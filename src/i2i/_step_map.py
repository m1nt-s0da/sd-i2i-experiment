import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ConfigI2IStepMap
from ._stepper import PredictNoise, _build_stepper


def compute_step_counts(
    input_latent: torch.Tensor,
    noise: torch.Tensor,
    *,
    scheduler,
    start_step: int,
    config: ConfigI2IStepMap,
    predict_noise: PredictNoise,
) -> torch.Tensor:
    """最後のprobe_stepsステップを試走したx0の推移から、latent pixelごとのi2iステップ数を返す。

    入力潜在に(steps - probe_steps)時点のノイズを加えた潜在から、最後のstepまで試走する。
    試走中にx0が大きく変化したpixelほど多くのステップ（start_stepから）を割り当て、
    ほとんど変化しないpixelは最後のrefine_stepsステップだけにする。
    probe_modeがrenoiseなら、各ステップで入力にノイズを加え直してx0を推定する（試走の軌跡を辿らない）。
    変化量はconfig.metricで選ぶ（step_delta: 連続するstep間のx0差、input_distance: 入力潜在とx0の差）。形状は(B, 1, H, W)のint64。
    """
    stepper = _build_stepper(scheduler)
    steps = len(scheduler.timesteps)
    remaining = steps - start_step
    probe_steps = min(config.probe_steps, remaining)
    refine_steps = min(config.refine_steps, remaining)

    x0s = []
    current = stepper.add_noise(input_latent, noise, steps - probe_steps).to(
        input_latent.dtype
    )
    for index in tqdm(range(steps - probe_steps, steps), desc="step_map probe"):
        if config.probe_mode == "renoise":
            current = stepper.add_noise(input_latent, noise, index).to(
                input_latent.dtype
            )
        x0, current = stepper.evaluate(current, index, predict_noise)
        x0s.append(x0)

    if config.metric == "input_distance":
        pairs = [(input_latent, x0) for x0 in x0s]
    else:
        pairs = list(zip(x0s, x0s[1:]))

    changed = torch.zeros(
        (input_latent.shape[0], 1, *input_latent.shape[-2:]), device=input_latent.device
    )
    for before, after in pairs:
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

    fraction = changed / max(1, len(pairs))
    counts = refine_steps + torch.round(fraction * (remaining - refine_steps)).long()
    tqdm.write(
        f"[step_map] probed {probe_steps} step(s): per-pixel steps "
        f"min {counts.min().item()}, mean {counts.float().mean().item():.1f}, "
        f"max {counts.max().item()} (of {remaining})"
    )
    return counts
