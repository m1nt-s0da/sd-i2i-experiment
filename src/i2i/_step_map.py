import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ConfigI2IStepMap
from ._stepper import PredictNoise, _build_stepper


def compute_step_counts(
    latents: torch.Tensor,
    *,
    scheduler,
    start_step: int,
    config: ConfigI2IStepMap,
    predict_noise: PredictNoise,
) -> torch.Tensor:
    """数ステップ試走したx0の推移から、latent pixelごとのi2iステップ数を返す。

    試走中にx0が大きく変化したpixelほど多くのステップ（start_stepから）を割り当て、
    ほとんど変化しないpixelは最後のrefine_stepsステップだけにする。形状は(B, 1, H, W)のint64。
    """
    stepper = _build_stepper(scheduler)
    remaining = len(scheduler.timesteps) - start_step
    probe_steps = min(config.probe_steps, remaining)
    refine_steps = min(config.refine_steps, remaining)

    x0s = []
    current = latents
    for index in tqdm(
        range(start_step, start_step + probe_steps), desc="step_map probe"
    ):
        x0, current = stepper.evaluate(current, index, predict_noise)
        x0s.append(x0)

    changed = torch.zeros((latents.shape[0], 1, *latents.shape[-2:]), device=latents.device)
    for before, after in zip(x0s, x0s[1:]):
        delta = torch.linalg.vector_norm((after - before).float(), dim=1)[:, None]
        if config.pool_kernel > 1:
            delta = F.avg_pool2d(
                delta,
                config.pool_kernel,
                stride=1,
                padding=config.pool_kernel // 2,
                count_include_pad=False,
            )
        changed += (delta / delta.mean().clamp_min(1e-8) > config.threshold).float()

    fraction = changed / max(1, len(x0s) - 1)
    counts = refine_steps + torch.round(fraction * (remaining - refine_steps)).long()
    tqdm.write(
        f"[step_map] probed {probe_steps} step(s): per-pixel steps "
        f"min {counts.min().item()}, mean {counts.float().mean().item():.1f}, "
        f"max {counts.max().item()} (of {remaining})"
    )
    return counts
