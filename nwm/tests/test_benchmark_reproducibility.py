import torch

from diffusion import create_gaussian_diffusion
from scripts.benchmark_reproducibility import (
    expand_sample_keys,
    samplewise_noise_schedule,
    samplewise_randn,
    stable_seed,
)


def test_stable_seed_uses_the_full_semantic_identity() -> None:
    assert stable_seed(0, "recon", "direct", 7) == stable_seed(
        0, "recon", "direct", 7
    )
    assert stable_seed(0, "recon", "direct", 7) != stable_seed(
        0, "recon", "direct", 8
    )
    assert stable_seed(0, "recon", "direct", 7) != stable_seed(
        0, "scand", "direct", 7
    )


def test_samplewise_noise_is_batch_partition_invariant() -> None:
    keys = ["0", "1", "2", "3", "4"]
    together = samplewise_randn(
        keys,
        (2, 3),
        base_seed=17,
        stream="unit-test",
        device="cpu",
    )
    partitioned = torch.cat(
        [
            samplewise_randn(
                keys[:2],
                (2, 3),
                base_seed=17,
                stream="unit-test",
                device="cpu",
            ),
            samplewise_randn(
                keys[2:],
                (2, 3),
                base_seed=17,
                stream="unit-test",
                device="cpu",
            ),
        ]
    )
    torch.testing.assert_close(together, partitioned, rtol=0, atol=0)


def test_samplewise_schedule_is_batch_partition_invariant() -> None:
    keys = [10, 11, 12, 13]
    together = samplewise_noise_schedule(
        keys,
        (2, 2),
        draws=5,
        base_seed=3,
        stream="rollout/step=4",
        device="cpu",
    )
    left = samplewise_noise_schedule(
        keys[:1],
        (2, 2),
        draws=5,
        base_seed=3,
        stream="rollout/step=4",
        device="cpu",
    )
    right = samplewise_noise_schedule(
        keys[1:],
        (2, 2),
        draws=5,
        base_seed=3,
        stream="rollout/step=4",
        device="cpu",
    )
    torch.testing.assert_close(together, torch.cat((left, right), dim=1), rtol=0, atol=0)


def test_expand_sample_keys_distinguishes_goals() -> None:
    assert expand_sample_keys([4, 9], 2) == [
        "4/goal=0",
        "4/goal=1",
        "9/goal=0",
        "9/goal=1",
    ]


def test_complete_ddpm_chain_is_batch_partition_invariant() -> None:
    diffusion = create_gaussian_diffusion(
        timestep_respacing="4",
        diffusion_steps=1000,
        learn_sigma=False,
    )
    keys = ["sample-0", "sample-1", "sample-2"]

    def run(selected: list[str]) -> torch.Tensor:
        schedule = samplewise_noise_schedule(
            selected,
            (1, 2, 2),
            draws=diffusion.num_timesteps + 1,
            base_seed=91,
            stream="ddpm-test",
            device="cpu",
        )

        def zero_model(x: torch.Tensor, timesteps: torch.Tensor, **_: object) -> torch.Tensor:
            del timesteps
            return torch.zeros_like(x)

        return diffusion.p_sample_loop(
            zero_model,
            (len(selected), 1, 2, 2),
            noise=schedule[0],
            step_noises=schedule[1:],
            device=torch.device("cpu"),
        )

    together = run(keys)
    partitioned = torch.cat((run(keys[:1]), run(keys[1:])), dim=0)
    torch.testing.assert_close(together, partitioned, rtol=0, atol=0)
