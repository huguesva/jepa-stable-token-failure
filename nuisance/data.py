"""Dataset plumbing for the nuisance experiment.

Wraps ``stable_worldmodel``'s HDF5 dataset so that the nuisance border is
painted on the raw uint8 frames *before* the usual ImageNet preprocessing, using
the episode index that the loader already has in hand.  Nothing about the task
content, actions, proprioception, rewards or the train/val split is touched.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import stable_worldmodel as swm

from nuisance.nuisance import NuisanceRenderer


class NuisanceHDF5Dataset(swm.data.HDF5Dataset):
    """HDF5 dataset that paints an episode-indexed nuisance border on ``pixels``.

    The base class applies ``self.transform`` at the end of ``_load_slice``; we
    keep that slot empty and hold the user pipeline in ``post_transform`` so the
    nuisance is applied to raw uint8 frames first.

    Args:
        renderer: nuisance renderer, or ``None`` for an untouched dataset.
        episode_offset: added to the episode index before seeding the nuisance,
            so that disjoint nuisance streams can be requested for the same data.
        **kwargs: forwarded to :class:`stable_worldmodel.data.HDF5Dataset`.
    """

    def __init__(
        self,
        *args,
        renderer: NuisanceRenderer | None = None,
        episode_offset: int = 0,
        transform: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, transform=None, **kwargs)
        self.renderer = renderer
        self.episode_offset = episode_offset
        self.post_transform = transform

    def frame_index(self, start: int, end: int, n: int) -> torch.Tensor:
        """Indices handed to the renderer, in the unit the spec asks for.

        The loader walks *raw* dataset frames and subsamples by ``frameskip``,
        while the evaluation policy counts *model* frames.  A persistence
        parameter defined per model frame is only the same quantity in both
        places if the loader converts, which ``frame_unit='model'`` requests.
        """
        idx = torch.arange(start, end, self.frameskip)[:n]
        if self.renderer is not None and self.renderer.spec.frame_unit == "model":
            idx = idx // self.frameskip
        return idx

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        steps = super()._load_slice(ep_idx, start, end)

        if self.renderer is not None and "pixels" in steps:
            pixels = steps["pixels"]
            frame_index = self.frame_index(start, end, pixels.shape[0])
            steps["pixels"] = self.renderer.render(
                pixels, int(ep_idx) + self.episode_offset, frame_index
            )

        return self.post_transform(steps) if self.post_transform else steps


def limit_to_episodes(dataset: swm.data.Dataset, max_episodes: int | None) -> None:
    """Restrict a dataset in place to its first ``max_episodes`` episodes.

    Used to trade dataset size for wall-clock during pilots.  Applied identically
    to every condition so it can never favour one of them.
    """
    if max_episodes is None or max_episodes >= len(dataset.lengths):
        return
    dataset.clip_indices = [(ep, s) for ep, s in dataset.clip_indices if ep < max_episodes]
