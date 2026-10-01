"""Task-irrelevant visual nuisance modes for LeWM observations.

The nuisance is a coloured pattern painted into a *border* region of the frame.
Two canvas modes are available and both leave the task pixels unoccluded:

``pad``     (default) the canvas grows to ``H + 2b`` and the scene is pasted in
            at full native resolution.  Nothing about the scene changes, so the
            matched-geometry control costs almost nothing.
``shrink``  the scene is downscaled into ``(H - 2b, W - 2b)`` and pasted onto a
            canvas of the original size.  Keeps the input size fixed but costs
            scene resolution, which on PushT turned out to be expensive.

In both cases the scene pixels are identical across conditions and only the
border content differs:

``neutral``     constant mid-grey border (matched-geometry control / baseline)
``persistent``  border drawn from an episode-level latent, constant within episode
``fast``        border latent resampled every frame from the *same* distribution
``markov``      border latent following a stationary first-order refresh chain
                with retention probability ``rho``; ``rho=1`` reproduces
                ``persistent`` and ``rho=0`` reproduces ``fast`` exactly
``clean``       no modification at all (native resolution reference)

Two *geometries* decide where the colour goes, i.e. how many ViT tokens it can
possibly reach:

``tile``    the perimeter is cut into ``n_tiles`` arcs, each painted its own
            colour, over a ring of ``ring`` pixels.  This sweeps pixel area but
            **not** token count: every arc, however thin, still lands inside the
            outer ring of patches, so all 68 boundary tokens of a 252px input are
            contaminated at every ring width.  That is why the pixel-area sweep
            of the first campaign came back flat.
``patch``   ``n_patches`` *complete* patch cells of the padded border are each
            painted a single colour and the rest of the border stays grey.  The
            token footprint is then exactly ``n_patches`` out of ``(S/p)**2``,
            which is the knob the earlier design never had.

The latent ``z`` has ``3 * n_tiles`` (tile) or ``3 * n_patches`` (patch)
dimensions and its ground truth is available for held-out probing.

Randomness is derived from a counter-based hash of
``(base_seed, stream, episode_id, anchor)`` so the exact same nuisance is
reproduced in training, evaluation and probing, in any worker process, without
carrying RNG state around.  For ``markov`` the anchor is the most recent refresh
time of a stationary Bernoulli(1 - rho) refresh process whose indicators are
themselves a deterministic function of ``(base_seed, stream, episode_id)``; the
marginal law of ``z_t`` is therefore uniform for every ``rho``, and
``Corr(z_t, z_{t+m}) = rho**m``.

``frame_unit`` says what the frame indices handed to the renderer mean.  The
evaluation policy counts *model* frames, while the HDF5 loader counts *raw*
frames and subsamples by ``frameskip``; the temporal axis of the nuisance is
only comparable between the two when both are expressed in model frames, so the
persistence campaign sets ``frame_unit='model'`` and the loader divides.  The
legacy default ``'raw'`` reproduces the first campaign bit-for-bit.

Everything is PyTorch; no NumPy is used for the numerical work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_MASK64 = (1 << 64) - 1

#: Conditions that paint a border (i.e. use the matched padded geometry).
BORDER_CONDITIONS = ("neutral", "persistent", "fast", "markov")
#: All supported conditions.
CONDITIONS = ("clean",) + BORDER_CONDITIONS
#: Supported spatial layouts of the nuisance colour.
GEOMETRIES = ("tile", "patch")
#: Units the caller's ``frame_index`` may be expressed in.
FRAME_UNITS = ("raw", "model")

#: Salt separating the refresh-indicator stream from the colour-draw stream.
_REFRESH_SALT = 0x5245465253480001
#: Refresh indicators are materialised in one fixed-length block per episode so
#: that the sequence never depends on how many frames a caller happens to ask
#: for.  PushT episodes are ~125 raw / ~25 model frames, so this is ample.
_REFRESH_LEN = 4096


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit integer mix (no dependence on Python's hash seed)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def derive_seed(*parts: int) -> int:
    """Mix an arbitrary tuple of integers into one 63-bit seed."""
    h = 0
    for p in parts:
        h = _splitmix64(h ^ (int(p) & _MASK64))
    return h & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class NuisanceSpec:
    """Configuration of the nuisance renderer.

    Attributes:
        condition: one of :data:`CONDITIONS`.
        border: border width in pixels on each side of the (square) frame.
        geometry: ``tile`` (perimeter arcs, legacy) or ``patch`` (whole patch
            cells of the padded border, so the token footprint is exact).
        n_tiles: number of independently coloured arcs around the perimeter,
            used by the ``tile`` geometry.  Latent dimension ``3 * n_tiles``.
        n_patches: number of whole patch cells painted by the ``patch``
            geometry.  Latent dimension ``3 * n_patches``.
        patch_size: ViT patch size the ``patch`` geometry aligns to.  Must
            divide ``border`` so that no painted cell can touch a scene pixel.
        amplitude: fraction of the full colour range spanned by the nuisance.
            ``1.0`` gives colours uniform on ``[0, 255]``; ``0.5`` gives colours
            uniform on ``[64, 192]``.  Controls contrast, not area.
        persistence: number of frames a latent is held for, used by the legacy
            ``persistent``/``fast`` conditions.  ``0`` means "hold for the whole
            episode", ``1`` means "resample every frame".
        rho: retention probability of the ``markov`` condition.  The latent is
            kept from the previous frame with probability ``rho`` and redrawn
            from the same distribution otherwise, so the per-frame marginal is
            identical for every ``rho`` and only the temporal correlation
            ``rho**m`` changes.  ``1`` is episode-stable, ``0`` is i.i.d.
        frame_unit: whether the ``frame_index`` handed to the renderer counts
            raw dataset frames (legacy) or model frames (the unit the temporal
            correlation is defined in).  The dataset honours this; the renderer
            itself always receives indices already in the right unit.
        base_seed: root seed of the nuisance stream.
        stream: separates independent nuisance streams (e.g. train vs eval).
    """

    condition: str = "clean"
    mode: str = "pad"
    border: int = 14
    ring: int = 0
    geometry: str = "tile"
    n_tiles: int = 16
    n_patches: int = 4
    patch_size: int = 14
    amplitude: float = 1.0
    persistence: int = 0
    rho: float = 1.0
    frame_unit: str = "raw"
    base_seed: int = 20260922
    stream: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}, expected one of {CONDITIONS}")
        if self.mode not in ("pad", "shrink"):
            raise ValueError(f"mode must be 'pad' or 'shrink', got {self.mode!r}")
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {self.geometry!r}")
        if self.frame_unit not in FRAME_UNITS:
            raise ValueError(f"frame_unit must be one of {FRAME_UNITS}, got {self.frame_unit!r}")
        if not 0 <= self.ring <= self.border:
            raise ValueError(f"ring must be in [0, border]={self.border}, got {self.ring}")
        if self.condition == "fast" and self.persistence != 1:
            object.__setattr__(self, "persistence", 1)
        if self.condition == "persistent" and self.persistence < 0:
            raise ValueError("persistence must be >= 0")
        if self.condition == "markov" and not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {self.rho}")
        if self.geometry == "patch":
            if self.mode != "pad":
                raise ValueError("the 'patch' geometry requires mode='pad'")
            if self.patch_size <= 0 or self.border % self.patch_size:
                raise ValueError(
                    f"border {self.border} must be a positive multiple of patch_size "
                    f"{self.patch_size} for the 'patch' geometry"
                )
            if self.border != self.patch_size:
                raise ValueError(
                    "only a one-patch-deep border is supported so far "
                    f"(border={self.border}, patch_size={self.patch_size})"
                )
            if self.n_patches < 1:
                raise ValueError(f"n_patches must be >= 1, got {self.n_patches}")

    @property
    def latent_dim(self) -> int:
        return 3 * (self.n_patches if self.geometry == "patch" else self.n_tiles)

    @property
    def paints_border(self) -> bool:
        return self.condition in BORDER_CONDITIONS

    @property
    def is_stochastic(self) -> bool:
        """Whether the border content actually varies (``neutral`` does not)."""
        return self.condition in ("persistent", "fast", "markov")

    @property
    def ring_width(self) -> int:
        """Width of the painted ring; ``ring=0`` means "the whole border"."""
        return self.border if self.ring == 0 else self.ring

    @property
    def retention(self) -> float:
        """One-step probability that the latent is *not* redrawn.

        Defined for every condition so that the persistence axis has a single
        name: ``persistent`` is 1, ``fast`` is 0, ``markov`` is ``rho``.
        """
        if self.condition == "markov":
            return float(self.rho)
        if self.condition == "persistent":
            return 1.0
        if self.condition == "fast":
            return 0.0
        return float("nan")

    def output_size(self, input_size: int) -> int:
        """Model-input resolution produced from a raw frame of ``input_size``."""
        if not self.paints_border or self.mode == "shrink":
            return input_size
        return input_size + 2 * self.border

    def border_area_fraction(self, input_size: int) -> float:
        """Fraction of the model input occupied by the whole border."""
        if not self.paints_border:
            return 0.0
        out = self.output_size(input_size)
        inner = input_size if self.mode == "pad" else input_size - 2 * self.border
        return 1.0 - (inner / out) ** 2

    def painted_area_fraction(self, input_size: int) -> float:
        """Fraction of the model input actually carrying nuisance colour."""
        if not self.is_stochastic:
            return 0.0
        out = self.output_size(input_size)
        if self.geometry == "patch":
            return self.n_patches * self.patch_size**2 / out**2
        r = self.ring_width
        return 1.0 - ((out - 2 * r) / out) ** 2

    def token_grid(self, input_size: int) -> int:
        """Number of patch tokens per side of the model input."""
        out = self.output_size(input_size)
        if out % self.patch_size:
            raise ValueError(f"input {out} is not a multiple of patch_size {self.patch_size}")
        return out // self.patch_size

    def painted_token_count(self, input_size: int) -> int:
        """Patch tokens that carry nuisance colour.

        This is the quantity the first campaign never controlled: for the
        ``tile`` geometry it is the *whole* boundary ring regardless of how thin
        the painted arc is, while for ``patch`` it is exactly ``n_patches``.
        """
        if not self.is_stochastic:
            return 0
        g = self.token_grid(input_size)
        if self.geometry == "patch":
            return min(self.n_patches, g * g - (g - 2) ** 2)
        # a tile arc of any width >= 1 still lands in the outermost patch ring
        depth = min(g, -(-self.ring_width // self.patch_size))
        return g * g - max(0, g - 2 * depth) ** 2


def _perimeter_tile_map(size: int, border: int, n_tiles: int, device: torch.device) -> torch.Tensor:
    """Map every border pixel of a ``size x size`` frame to a tile index.

    Interior pixels get index ``-1``.  The perimeter is walked clockwise
    starting at the top-left corner and cut into ``n_tiles`` equal arcs.
    """
    ii = torch.arange(size, device=device).view(-1, 1).expand(size, size)
    jj = torch.arange(size, device=device).view(1, -1).expand(size, size)

    top = ii < border
    bottom = ii >= size - border
    left = jj < border
    right = jj >= size - border
    is_border = top | bottom | left | right

    inner = size - 2 * border
    # clockwise perimeter coordinate; corners are attributed to the top/bottom runs
    p = torch.zeros(size, size, dtype=torch.long, device=device)
    p = torch.where(top, jj, p)
    p = torch.where(bottom & ~top, size + inner + (size - 1 - jj), p)
    side = ~top & ~bottom
    p = torch.where(side & right, size + (ii - border), p)
    p = torch.where(side & left & ~right, 2 * size + inner + (size - border - 1 - ii), p)

    perimeter = 2 * size + 2 * inner
    tile = (p.to(torch.float64) * n_tiles / perimeter).floor().to(torch.long).clamp_(0, n_tiles - 1)
    return torch.where(is_border, tile, torch.full_like(tile, -1))


def _patch_cell_map(
    size: int, patch_size: int, n_patches: int, device: torch.device
) -> torch.Tensor:
    """Map every pixel of a ``size x size`` frame to a painted-patch index.

    Only whole patch cells of the outer ring are painted, so the set of ViT
    tokens the nuisance can reach is exactly the set of selected cells.  Pixels
    of unpainted cells get ``-1``.
    """
    if size % patch_size:
        raise ValueError(f"size {size} is not a multiple of patch_size {patch_size}")
    grid = size // patch_size
    cell = torch.full((size, size), -1, dtype=torch.long, device=device)
    for j, (r, c) in enumerate(select_patch_cells(grid, n_patches)):
        cell[
            r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size
        ] = j
    return cell


def boundary_patch_walk(grid: int) -> list[tuple[int, int]]:
    """Clockwise walk of the outer ring of a ``grid x grid`` patch lattice.

    Starts at the top-left cell and returns ``4 * grid - 4`` coordinates.  The
    walk fixes a canonical ordering so that "the ``n`` patches used at this
    dose" is reproducible from ``n`` alone.
    """
    if grid < 2:
        raise ValueError(f"grid must be >= 2, got {grid}")
    cells = [(0, c) for c in range(grid)]
    cells += [(r, grid - 1) for r in range(1, grid)]
    cells += [(grid - 1, c) for c in range(grid - 2, -1, -1)]
    cells += [(r, 0) for r in range(grid - 2, 0, -1)]
    return cells


def select_patch_cells(grid: int, n_patches: int) -> list[tuple[int, int]]:
    """Pick ``n_patches`` boundary cells spread evenly around the ring."""
    ring = boundary_patch_walk(grid)
    n = min(n_patches, len(ring))
    return [ring[(j * len(ring)) // n] for j in range(n)]


@dataclass
class NuisanceRenderer:
    """Applies a :class:`NuisanceSpec` to uint8 frame stacks."""

    spec: NuisanceSpec
    _tile_cache: dict = field(default_factory=dict, repr=False)
    _anchor_cache: dict = field(default_factory=dict, repr=False)

    # -- latents ---------------------------------------------------------

    def block_index(self, frame_index: torch.Tensor | int) -> torch.Tensor | int:
        """Frame index -> latent block index (constant within a block)."""
        if self.spec.persistence <= 0:
            return frame_index * 0 if torch.is_tensor(frame_index) else 0
        return frame_index // self.spec.persistence

    def _anchor_table(self, episode_id: int) -> torch.Tensor:
        """Last refresh time at or before each index, for the ``markov`` chain.

        The refresh indicators are i.i.d. Bernoulli(1 - rho) draws generated in
        one fixed-length block per episode, so the table never depends on how
        many frames a caller asks for.  Index 0 always refreshes, which makes
        the chain start from its stationary law at the first frame.
        """
        table = self._anchor_cache.get(episode_id)
        if table is not None:
            return table
        gen = torch.Generator()
        gen.manual_seed(
            derive_seed(self.spec.base_seed, self.spec.stream, episode_id, _REFRESH_SALT)
        )
        u = torch.rand(_REFRESH_LEN, generator=gen)
        refresh = u >= self.spec.rho          # rho=1 -> never, rho=0 -> always
        refresh[0] = True
        idx = torch.arange(_REFRESH_LEN)
        table = torch.cummax(torch.where(refresh, idx, torch.full_like(idx, -1)), dim=0).values
        if len(self._anchor_cache) >= 4096:   # bounded per-worker memo
            self._anchor_cache.clear()
        self._anchor_cache[episode_id] = table
        return table

    def anchor_index(self, episode_id: int, frame_index: torch.Tensor) -> torch.Tensor:
        """Frame index -> index of the draw that is still in force at that frame."""
        if self.spec.condition != "markov":
            return self.block_index(frame_index)
        k = frame_index.long()
        if int(k.max()) >= _REFRESH_LEN:
            raise ValueError(
                f"frame index {int(k.max())} exceeds the refresh block length {_REFRESH_LEN}"
            )
        return self._anchor_table(episode_id)[k.clamp_(min=0)]

    def latent(self, episode_id: int, frame_index: torch.Tensor) -> torch.Tensor:
        """Ground-truth nuisance latent in ``[0, 1]`` for each frame.

        Args:
            episode_id: identifier of the episode the frames belong to.
            frame_index: 1-D long tensor of frame indices within the episode,
                expressed in the unit named by ``spec.frame_unit``.

        Returns:
            Tensor ``(T, latent_dim)`` of uniform ``[0, 1]`` factors.  For
            ``neutral``/``clean`` this is all ``0.5`` (no variation).
        """
        t = frame_index.numel()
        if not self.spec.is_stochastic:
            return torch.full((t, self.spec.latent_dim), 0.5)

        anchors = self.anchor_index(episode_id, frame_index)
        out = torch.empty(t, self.spec.latent_dim)
        gen = torch.Generator()
        for k, b in enumerate(anchors.tolist()):
            gen.manual_seed(derive_seed(self.spec.base_seed, self.spec.stream, episode_id, b))
            out[k] = torch.rand(self.spec.latent_dim, generator=gen)
        return out

    def tile_colors(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent in ``[0, 1]`` -> uint8-valued colours ``(T, n_cells, 3)``."""
        a = self.spec.amplitude
        vals = 255.0 * (0.5 + a * (latent - 0.5))
        return vals.view(latent.shape[0], -1, 3).clamp_(0.0, 255.0)

    # -- rendering -------------------------------------------------------

    def _tile_map(self, size: int, device: torch.device) -> torch.Tensor:
        """Pixel -> colour-cell index map; ``-1`` means "leave it grey"."""
        key = (size, device.type, device.index)
        if key not in self._tile_cache:
            if self.spec.geometry == "patch":
                self._tile_cache[key] = _patch_cell_map(
                    size, self.spec.patch_size, self.spec.n_patches, device
                )
            else:
                self._tile_cache[key] = _perimeter_tile_map(
                    size, self.spec.ring_width, self.spec.n_tiles, device
                )
        return self._tile_cache[key]

    def render(self, frames: torch.Tensor, episode_id: int, frame_index: torch.Tensor) -> torch.Tensor:
        """Apply the nuisance to a stack of frames.

        Args:
            frames: uint8 tensor ``(T, C, H, W)`` with ``H == W``.
            episode_id: episode identifier driving the nuisance RNG.
            frame_index: 1-D long tensor ``(T,)`` of frame indices in the episode.

        Returns:
            uint8 tensor ``(T, C, S, S)`` where ``S = spec.output_size(H)``.
        """
        if not self.spec.paints_border:
            return frames

        if frames.ndim != 4:
            raise ValueError(f"expected (T, C, H, W), got {tuple(frames.shape)}")
        t, c, h, w = frames.shape
        if h != w:
            raise ValueError(f"expected square frames, got {h}x{w}")
        b = self.spec.border
        if self.spec.mode == "shrink" and 2 * b >= h:
            raise ValueError(f"border {b} too large for size {h}")

        size = self.spec.output_size(h)

        # 1. place the scene, leaving the ring free. `pad` keeps every task pixel
        #    at native resolution; `shrink` trades resolution for a fixed input size.
        if self.spec.mode == "pad":
            scene = frames.float()
        else:
            scene = F.interpolate(
                frames.float(), size=(h - 2 * b, w - 2 * b), mode="bilinear", align_corners=False
            )

        # 2. paint the ring
        canvas = torch.full((t, c, size, size), 128.0, device=frames.device, dtype=torch.float32)
        if self.spec.is_stochastic:
            tile_map = self._tile_map(size, frames.device)        # (S, S) in [-1, n_cells)
            colors = self.tile_colors(self.latent(episode_id, frame_index)).to(frames.device)
            painted = colors[:, tile_map.clamp(min=0)].permute(0, 3, 1, 2)   # (T, C, S, S)
            # cells not selected by the geometry stay grey, so the painted area
            # (tile) or token footprint (patch) can be swept at fixed geometry
            canvas = torch.where(tile_map.view(1, 1, size, size) >= 0, painted, canvas)

        # 3. paste the scene into the interior
        canvas[:, :, b : size - b, b : size - b] = scene
        return canvas.round_().clamp_(0, 255).to(frames.dtype)


def make_renderer(**kwargs) -> NuisanceRenderer:
    """Convenience constructor: ``make_renderer(condition="persistent", border=24)``."""
    return NuisanceRenderer(NuisanceSpec(**kwargs))
