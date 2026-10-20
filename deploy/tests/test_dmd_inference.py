"""DMD sampling semantics and deployment CLI contracts; no robot interfaces."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nwm"))

import torch
from dmd_sampler import DMDInferenceSampler
from jetson_infer import (
    MODEL_CHOICES, add_sampler_arguments, validate_sampler_arguments,
    checkpoint_relative_path, resolve_sampling,
)


class Arguments(unittest.TestCase):
    def parse(self, argv, default_steps=50):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=MODEL_CHOICES, default="opennwm")
        parser.add_argument("--steps", type=int)
        add_sampler_arguments(parser, default_steps=default_steps)
        args = parser.parse_args(argv)
        validate_sampler_arguments(parser, args)
        return args

    def test_legacy_defaults_and_explicit_ddim_remain(self):
        for count in (50, 250):
            args = self.parse([], count)
            self.assertEqual((args.sampler, args.steps), ("ddpm", count))
        args = self.parse(["--model", "nwm-b", "--sampler", "ddim", "--steps", "10"])
        self.assertEqual((args.sampler, args.steps), ("ddim", 10))

    def test_both_student_aliases_and_sampler_selection(self):
        for name, directory in (("opennwm-dmd", "OpenNWM-DMD"), ("nwm-dmd", "NWM-DMD")):
            args = self.parse(["--model", name])
            self.assertEqual((args.sampler, args.steps), ("dmd", 4))
            self.assertEqual(checkpoint_relative_path(name, args.sampler), directory + "/student.pt")
        for name in ("opennwm", "nwm-b"):
            args = self.parse(["--model", name, "--sampler", "dmd"])
            self.assertEqual(args.steps, 4)
            self.assertIn("DMD/student.pt", checkpoint_relative_path(name, args.sampler))

    def test_reject_wrong_steps_sampler_or_eta_before_loading(self):
        for argv in (
            ["--model", "opennwm-dmd", "--steps", "10"],
            ["--model", "nwm-dmd", "--sampler", "ddpm"],
            ["--sampler", "dmd", "--ddim-eta", "0"],
            ["--sampler", "ddpm", "--ddim-eta", "0"],
            ["--sampler", "ddim", "--ddim-eta", "nan"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(argv)
        with self.assertRaises(ValueError):
            resolve_sampling("opennwm-dmd", 250, "ddim")


class Schedule(unittest.TestCase):
    def setUp(self):
        self.alpha = torch.cumprod(1 - torch.linspace(.0001, .02, 1000), dim=0)
        self.sampler = DMDInferenceSampler(self.alpha, "cpu")
        self.shape = (2, 4, 3, 3)

    def test_release_equations_original_timesteps_no_clipping_or_sigma(self):
        noise = torch.randn(self.shape)
        draws = [torch.randn(self.shape) for _ in range(4)]
        calls = []

        def model(x, t, offset):
            calls.append(t.tolist())
            eps = x * .2 + offset
            return torch.cat((eps, torch.full_like(eps, 1e6)), dim=1)

        actual = self.sampler.p_sample_loop(model, self.shape, noise=noise,
                                           step_noises=draws, model_kwargs={"offset": .03})
        expected = noise
        for i, t in enumerate((999, 749, 499, 249)):
            a = self.alpha[t]
            clean = (expected - (1 - a).sqrt() * (expected * .2 + .03)) / a.sqrt()
            if i < 3:
                b = self.alpha[(749, 499, 249)[i]]
                expected = b.sqrt() * clean + (1 - b).sqrt() * draws[i]
        torch.testing.assert_close(actual, clean, rtol=1e-6, atol=1e-6)
        self.assertEqual(calls, [[t, t] for t in (999, 749, 499, 249)])
        self.assertGreater(actual.abs().max().item(), 1)

    def test_samplewise_noise_keeps_candidate_chunking_invariant(self):
        from scripts.benchmark_reproducibility import samplewise_noise_schedule
        keys = [7, 7, 12]

        def run(keys):
            noise = samplewise_noise_schedule(keys, (4, 3, 3), draws=5,
                                              base_seed=123, stream="test/dmd", device="cpu")
            return self.sampler.p_sample_loop(
                lambda x, t: torch.cat((x * .1, x * 0), dim=1),
                (len(keys), 4, 3, 3), noise=noise[0], step_noises=noise[1:])

        together = run(keys)
        separate = torch.cat([run([k]) for k in keys])
        torch.testing.assert_close(together, separate, rtol=0, atol=0)
        torch.testing.assert_close(together[0], together[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(together[0], together[2]))

    def test_reject_guidance_clipping_and_wrong_noise_shape(self):
        for kwargs in ({"clip_denoised": True}, {"cond_fn": lambda: None},
                       {"step_noises": [torch.zeros(self.shape)]},
                       {"noise": torch.zeros(1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sampler.p_sample_loop(lambda x, t: x, self.shape, **kwargs)
        with self.assertRaises(ValueError):
            DMDInferenceSampler(self.alpha[:4], "cpu")


if __name__ == "__main__":
    unittest.main()
