from collections.abc import Callable
from typing import Protocol

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

# (スケーリング済みモデル入力, タイムステップ) -> ガイダンス適用後のノイズ予測
PredictNoise = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _Stepper(Protocol):
    def add_noise(
        self, input_latent: torch.Tensor, noise: torch.Tensor, index: int
    ) -> torch.Tensor:
        """入力潜在に、index時点のノイズ量のノイズを加える。"""
        ...

    def evaluate(
        self, latents: torch.Tensor, index: int, predict_noise: PredictNoise
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """予測x0と、index + 1時点の潜在表現を返す。"""
        ...


class _EulerStepper:
    def __init__(self, scheduler: EulerDiscreteScheduler) -> None:
        self._timesteps = scheduler.timesteps
        self._sigmas = [float(sigma) for sigma in scheduler.sigmas]

    def add_noise(self, input_latent, noise, index):
        return input_latent + self._sigmas[index] * noise

    def evaluate(self, latents, index, predict_noise):
        sigma = self._sigmas[index]
        model_input = latents / (sigma**2 + 1) ** 0.5
        noise_prediction = predict_noise(model_input, self._timesteps[index])
        x0 = latents - sigma * noise_prediction
        next_latents = latents + (self._sigmas[index + 1] - sigma) * noise_prediction
        return x0, next_latents


class _DDIMStepper:
    def __init__(self, scheduler: DDIMScheduler) -> None:
        self._timesteps = scheduler.timesteps
        alphas_cumprod = scheduler.alphas_cumprod
        self._alphas = [float(alphas_cumprod[int(t)]) for t in scheduler.timesteps]
        self._alphas.append(float(scheduler.final_alpha_cumprod))

    def add_noise(self, input_latent, noise, index):
        alpha = self._alphas[index]
        return alpha**0.5 * input_latent + (1 - alpha) ** 0.5 * noise

    def evaluate(self, latents, index, predict_noise):
        alpha, next_alpha = self._alphas[index], self._alphas[index + 1]
        noise_prediction = predict_noise(latents, self._timesteps[index])
        x0 = (latents - (1 - alpha) ** 0.5 * noise_prediction) / alpha**0.5
        next_latents = next_alpha**0.5 * x0 + (1 - next_alpha) ** 0.5 * noise_prediction
        return x0, next_latents


def _build_stepper(scheduler) -> _Stepper:
    if isinstance(scheduler, EulerDiscreteScheduler):
        return _EulerStepper(scheduler)
    if isinstance(scheduler, DDIMScheduler):
        return _DDIMStepper(scheduler)
    raise ValueError("step_map supports only the euler and ddim samplers.")
