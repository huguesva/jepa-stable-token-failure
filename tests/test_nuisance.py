from __future__ import annotations

import unittest

import torch

from nuisance.nuisance import NuisanceRenderer, NuisanceSpec


def spec(condition: str, rho: float = 1.0) -> NuisanceSpec:
    return NuisanceSpec(
        condition=condition,
        mode="pad",
        border=14,
        geometry="patch",
        n_patches=1,
        patch_size=14,
        amplitude=1.0,
        rho=rho,
        frame_unit="model",
        base_seed=20260922,
    )


class NuisanceRendererTest(unittest.TestCase):
    def test_patch_is_one_token_and_never_touches_the_scene(self) -> None:
        frames = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
        indices = torch.arange(4)
        grey = NuisanceRenderer(spec("neutral")).render(frames, 7, indices)
        stable = NuisanceRenderer(spec("markov", 1.0)).render(frames, 7, indices)

        self.assertEqual(grey.shape, (4, 3, 252, 252))
        self.assertEqual(stable.shape, grey.shape)
        self.assertTrue(torch.equal(grey[:, :, 14:238, 14:238], frames))
        self.assertTrue(torch.equal(stable[:, :, 14:238, 14:238], frames))

        changed = (stable != grey).any(dim=1)
        self.assertTrue(changed.flatten(1).sum(1).eq(14 * 14).all())
        self.assertFalse(changed[:, 14:, :].any())
        self.assertFalse(changed[:, :, 14:].any())

    def test_episode_stable_and_frame_random_timescales(self) -> None:
        frames = torch.zeros(6, 3, 224, 224, dtype=torch.uint8)
        indices = torch.arange(6)
        stable_renderer = NuisanceRenderer(spec("markov", 1.0))
        random_renderer = NuisanceRenderer(spec("markov", 0.0))

        stable = stable_renderer.render(frames, 11, indices)[:, :, :14, :14]
        stable_other_episode = stable_renderer.render(frames, 12, indices)[:, :, :14, :14]
        random = random_renderer.render(frames, 11, indices)[:, :, :14, :14]

        self.assertTrue(torch.equal(stable, stable[:1].expand_as(stable)))
        self.assertFalse(torch.equal(stable[0], stable_other_episode[0]))
        self.assertTrue(
            all(not torch.equal(random[t], random[t - 1]) for t in range(1, len(random)))
        )

    def test_renderer_is_deterministic(self) -> None:
        frames = torch.zeros(5, 3, 224, 224, dtype=torch.uint8)
        indices = torch.arange(5)
        renderer = NuisanceRenderer(spec("markov", 0.0))
        first = renderer.render(frames, 3, indices)
        second = renderer.render(frames, 3, indices)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
