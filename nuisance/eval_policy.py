"""Nuisance-aware wrapper around a planning policy.

At evaluation time the frames come from the live environment while the goal
image comes from the dataset.  This wrapper paints the nuisance border on both,
immediately before the policy sees them, so the transformation applied at test
time is exactly the one applied during training.

Each of the ``num_envs`` parallel environments runs exactly one episode in
``World.evaluate_from_dataset``, so "per-env" and "per-episode" coincide: the
nuisance latent of env ``e`` is drawn from the renderer's stream at episode id
``e``.  A dedicated stream id keeps evaluation nuisances disjoint from the ones
seen during training.

Goal handling (see the goal-conditioning caveat in the protocol):

``matched``      the goal image carries the *same* latent as the current frame.
                 The border therefore contributes nothing to the planning cost
                 in any condition, so a performance drop cannot be an artefact
                 of an unreachable nuisance goal.
``independent``  the goal image carries its own latent, drawn from the same
                 distribution.  Separately labelled mechanism probe only.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy

from nuisance.nuisance import NuisanceRenderer

GOAL_MODES = ("matched", "independent")


class NuisanceEvalPolicy(BasePolicy):
    """Applies the nuisance to ``pixels`` and ``goal`` before delegating."""

    def __init__(
        self,
        inner: BasePolicy,
        renderer: NuisanceRenderer,
        goal_mode: str = "matched",
        goal_episode_offset: int = 100_000,
    ) -> None:
        super().__init__()
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        self.type = "nuisance"
        self.inner = inner
        self.renderer = renderer
        self.goal_mode = goal_mode
        self.goal_episode_offset = goal_episode_offset
        self.step_count = 0

    # -- plumbing --------------------------------------------------------

    def set_env(self, env) -> None:
        self.env = env
        self.inner.set_env(env)

    def set_seed(self, seed: int) -> None:
        if hasattr(self.inner, "set_seed"):
            self.inner.set_seed(seed)

    @property
    def solver(self):
        return getattr(self.inner, "solver", None)

    # -- nuisance --------------------------------------------------------

    def _render_stack(self, frames: np.ndarray, episode_id: int, frame_index: torch.Tensor) -> np.ndarray:
        """frames: (T, H, W, C) uint8 -> same, with the border painted."""
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        out = self.renderer.render(x, episode_id, frame_index)
        return out.permute(0, 2, 3, 1).contiguous().numpy()

    def _apply(self, arr: np.ndarray, is_goal: bool) -> np.ndarray:
        n_envs, n_hist = arr.shape[0], arr.shape[1]
        # `pad` mode enlarges the frame, so the output cannot be allocated
        # with empty_like.
        size = self.renderer.spec.output_size(arr.shape[2])
        out = np.empty((n_envs, n_hist, size, size, arr.shape[4]), dtype=arr.dtype)

        # history slot k corresponds to frame (step_count - (n_hist - 1 - k))
        obs_index = self.step_count - (n_hist - 1) + torch.arange(n_hist)
        obs_index.clamp_(min=0)

        for e in range(n_envs):
            if not is_goal:
                out[e] = self._render_stack(arr[e], e, obs_index)
            elif self.goal_mode == "matched":
                # same latent as the current frame, for every history slot
                idx = torch.full((n_hist,), int(obs_index[-1]))
                out[e] = self._render_stack(arr[e], e, idx)
            else:
                idx = torch.zeros(n_hist, dtype=torch.long)
                out[e] = self._render_stack(arr[e], e + self.goal_episode_offset, idx)
        return out

    def get_action(self, info_dict: dict, **kwargs):
        d = dict(info_dict)
        if self.renderer.spec.paints_border:
            if "pixels" in d:
                d["pixels"] = self._apply(np.asarray(d["pixels"]), is_goal=False)
            if "goal" in d:
                d["goal"] = self._apply(np.asarray(d["goal"]), is_goal=True)
        action = self.inner.get_action(d, **kwargs)
        self.step_count += 1
        return action
