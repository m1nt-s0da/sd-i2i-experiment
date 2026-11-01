"""step_mapの試走方式・指標・しきい値を、複数の入力セットで比較する。

入力画像と目標プロンプトだけで決まる手法（uniform / step_map）と、評価専用の参照
（t2i、元プロンプトとの差から作る「変わるべき領域」の信号）を分けて扱う。

実行: uv run python experiments/depth_study.py [--quick]
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import i2i._main as i2i_main
from i2i._stepper import _build_stepper
from i2i.config import load_config
from i2i.config._config import ConfigI2IStepMap

ROOT = Path.cwd()
OUT = ROOT / "output" / "experiments" / "study"
INPUTS = OUT / "inputs"
STEPS = 30
PROBE_STEPS = 15
REFINE_STEPS = 5
VAE_SCALE = 8

SCENARIOS = {
    "city": dict(
        config="demo.toml",
        input="output/input.png",
        depth="output/depth.png",
        guidance=8.0,
        src_prompt="(masterpiece, best quality, photo-realistic:1.3), a cinematic shot of a modern city street at night, wet asphalt with glowing puddles, reflection of city lights, street lamps, distant skyscrapers, passing cars with light trails, long exposure effect, volumetric lighting, 8k resolution, highly detailed",
        src_negative="(worst quality, low quality:1.4), daylight, sunny, bright, deformed, blurry, bad anatomy, logo, watermark, signature, text, monochrome, illustration, anime, 3d render",
        caption_src="a photo of a city street at night",
        caption_tgt="a photo of a city street in bright sunny daytime",
    ),
    "anime": dict(
        config="demo_anime.toml",
        input="output/input_anime.png",
        depth=None,
        guidance=6.0,
        src_prompt="masterwork, best quality, anime artwork, anime style, key visual, 1girl, solo, cute school girl, long black hair, straight bangs, brown eyes, smiling, looking at viewer, school uniform, sailor suit, upper body, classroom background, soft lighting, depth of field",
        src_negative="worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, monochrome, grayscale, photo, realistic, 3d, rendering, text, watermark, signature",
        caption_src="an anime girl with long dark hair",
        caption_tgt="an anime boy with short black hair in a black school uniform",
    ),
    "landscape": dict(
        config="demo_anime.toml",
        generate=dict(width=768, height=432, seed=7),
        depth=None,
        guidance=7.5,
        src_prompt="a photo of a green summer meadow with a wooden cabin, blue sky with white clouds, sunny, highly detailed",
        src_negative="snow, winter, low quality, blurry, watermark, text",
        tgt_prompt="a photo of a snowy winter landscape with a wooden cabin, heavy snowfall, overcast grey sky, snow covered ground, highly detailed",
        tgt_negative="green grass, summer, blue sky, low quality, blurry, watermark, text",
        caption_src="a photo of a green summer meadow",
        caption_tgt="a photo of a snowy winter landscape",
    ),
    "cat": dict(
        config="demo_anime.toml",
        generate=dict(width=512, height=512, seed=7),
        depth=None,
        guidance=7.5,
        src_prompt="a photo of a cat sitting on a sofa in a living room, highly detailed",
        src_negative="dog, low quality, blurry, watermark, text",
        tgt_prompt="a photo of a dog sitting on a sofa in a living room, highly detailed",
        tgt_negative="cat, kitten, low quality, blurry, watermark, text",
        caption_src="a photo of a cat",
        caption_tgt="a photo of a dog",
    ),
}

MODES = {"traj": "trajectory", "reno": "renoise"}
METRICS = {"step": "step_delta", "inp": "input_distance"}
THRESHOLDS = [0.7, 1.0, 1.3]
UNIFORM_STARTS = [6, 12, 18]


class Stop(Exception):
    pass


PLAN: dict = {}
PIPELINE = None
CONTROLNET = None
ORIG_COMPUTE = i2i_main.compute_step_counts


def hook(input_latent, noise, *, scheduler, start_step, config, predict_noise):
    if PLAN["kind"] == "capture":
        PLAN["cap"] = (input_latent, noise, scheduler, predict_noise)
        raise Stop
    counts = ORIG_COMPUTE(
        input_latent,
        noise,
        scheduler=scheduler,
        start_step=start_step,
        config=config,
        predict_noise=predict_noise,
    )
    PLAN["counts"] = counts.cpu()
    return counts


i2i_main.compute_step_counts = hook


def make_cfg(sc, *, prompt, negative, seed, start_step, output_path, step_map=None):
    cfg = load_config(str(ROOT / sc["config"]), "toml").i2i
    cfg.prompt = prompt
    cfg.negative_prompt = negative
    cfg.guidance_scale = sc["guidance"]
    cfg.seed = seed
    cfg.start_step = start_step
    cfg.latent_distance_plot_path = None
    cfg.x0_distance_plot_path = None
    cfg.output_path = output_path
    cfg.input.path = sc["input"]
    cfg.pipeline.build_pipeline = lambda: PIPELINE
    cfg.step_map = step_map
    cfg.controlnet = cfg.controlnet if sc["depth"] else []
    for control in cfg.controlnet:
        control.condition_image = sc["depth"]
        control.save_image = None
        control.build_controlnet = lambda **kwargs: CONTROLNET
    return cfg


def run(cfg, plan):
    global PLAN
    PLAN = plan
    try:
        i2i_main.run_image_generation(cfg)
    except Stop:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return PLAN


def pooled_delta(a, b):
    delta = torch.linalg.vector_norm((a - b).float(), dim=1)[:, None]
    return F.avg_pool2d(delta, 3, stride=1, padding=1, count_include_pad=False)


def reference_signal(sc, seed):
    """元プロンプトと目標プロンプトで、入力に同じノイズを加えたx0予測がどれだけ違うか（評価専用）。"""
    captured = {}
    for tag, prompt, negative in (
        ("src", sc["src_prompt"], sc["src_negative"]),
        ("tgt", sc["tgt_prompt"], sc["tgt_negative"]),
    ):
        cfg = make_cfg(
            sc,
            prompt=prompt,
            negative=negative,
            seed=seed,
            start_step=0,
            output_path=None,
            step_map=ConfigI2IStepMap(),
        )
        captured[tag] = run(cfg, {"kind": "capture"})["cap"]
    input_latent, noise, scheduler, predict_tgt = captured["tgt"]
    predict_src = captured["src"][3]
    stepper = _build_stepper(scheduler)
    deltas = []
    with torch.no_grad():
        for index in range(STEPS - PROBE_STEPS, STEPS):
            current = stepper.add_noise(input_latent, noise, index).to(input_latent.dtype)
            x0_tgt, _ = stepper.evaluate(current, index, predict_tgt)
            x0_src, _ = stepper.evaluate(current, index, predict_src)
            deltas.append(pooled_delta(x0_tgt, x0_src))
    signal = torch.stack(deltas).mean(dim=0).cpu()
    return signal / signal.mean().clamp_min(1e-8)


def clip_edit(model, processor, image, caption_src, caption_tgt):
    inputs = processor(
        text=[caption_src, caption_tgt], images=image, return_tensors="pt", padding=True
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
    return float(logits[1] - logits[0])


def upsample(mask_latent, size):
    mask = np.kron(mask_latent.astype(np.uint8), np.ones((VAE_SCALE, VAE_SCALE), np.uint8))
    return mask[: size[1], : size[0]].astype(bool)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def generate_input(sc, name):
    path = INPUTS / f"{name}.png"
    if not path.exists():
        gen = sc["generate"]
        image = PIPELINE(
            prompt=sc["src_prompt"],
            negative_prompt=sc["src_negative"],
            width=gen["width"],
            height=gen["height"],
            num_inference_steps=STEPS,
            guidance_scale=sc["guidance"],
            generator=torch.Generator(device=PIPELINE.device).manual_seed(gen["seed"]),
        ).images[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    sc["input"] = str(path)


def condition_names():
    names = ["t2i"] + [f"u{k}" for k in UNIFORM_STARTS]
    for mode in MODES:
        for metric in METRICS:
            names += [f"{mode}_{metric}_t{t}" for t in THRESHOLDS]
    return names


def condition_plan(name):
    if name == "t2i":
        return 0, None
    if name.startswith("u"):
        return int(name[1:]), None
    mode, metric, thr = name.split("_")
    return 0, ConfigI2IStepMap(
        probe_steps=PROBE_STEPS,
        refine_steps=REFINE_STEPS,
        threshold=float(thr[1:]),
        pool_kernel=3,
        probe_mode=MODES[mode],
        metric=METRICS[metric],
    )


def montage(name, input_path, signal, depth_maps, names, path):
    maps = [n for n in names if n in depth_maps]
    rows = 2 + (len(maps) + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.4 * rows))
    for ax in axes.ravel():
        ax.axis("off")

    def show(ax, image, title, **kwargs):
        ax.imshow(image, **kwargs)
        ax.set_title(title, fontsize=9)

    show(axes[0, 0], Image.open(input_path), "input")
    show(axes[0, 1], signal[0, 0], "reference signal (eval only)", cmap="inferno")
    for ax, n in zip(axes[0, 2:], ("t2i", f"u{UNIFORM_STARTS[1]}")):
        show(ax, Image.open(OUT / f"{name}_s42_{n}.png"), n)
    for ax, n in zip(axes[1], [f"u{k}" for k in UNIFORM_STARTS][::2] + ["t2i"][:0]):
        show(ax, Image.open(OUT / f"{name}_s42_{n}.png"), n)
    for i, n in enumerate(maps):
        row, col = 2 + i // 2, (i % 2) * 2
        show(axes[row, col], Image.open(OUT / f"{name}_s42_{n}.png"), n)
        show(
            axes[row, col + 1],
            depth_maps[n][0, 0],
            f"{n} depth",
            cmap="inferno",
            vmin=0,
            vmax=STEPS,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    global PIPELINE, CONTROLNET
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [42] if args.quick else [42, 43, 44]
    scenario_names = ["cat"] if args.quick else list(SCENARIOS)
    OUT.mkdir(parents=True, exist_ok=True)

    base = load_config(str(ROOT / "demo.toml"), "toml").i2i
    PIPELINE = base.pipeline.build_pipeline()
    CONTROLNET = base.controlnet[0].build_controlnet(
        dtype=PIPELINE.unet.dtype, device=PIPELINE.device
    )
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    records = []
    for name in scenario_names:
        sc = SCENARIOS[name]
        if "generate" in sc:
            generate_input(sc, name)
        else:
            demo_cfg = load_config(str(ROOT / sc["config"]), "toml").i2i
            sc["tgt_prompt"] = demo_cfg.prompt
            sc["tgt_negative"] = demo_cfg.negative_prompt
        input_rgb = load_rgb(sc["input"])
        size = (input_rgb.shape[1], input_rgb.shape[0])
        first_seed_maps, first_signal = {}, None

        for seed in seeds:
            signal = reference_signal(sc, seed)
            if seed == seeds[0]:
                first_signal = signal
            flat = signal.flatten()
            high = (signal > torch.quantile(flat, 0.6))[0, 0].numpy()
            low = (signal < torch.quantile(flat, 0.4))[0, 0].numpy()
            high_px, low_px = upsample(high, size), upsample(low, size)

            for cond in condition_names():
                start_step, step_map = condition_plan(cond)
                image_path = OUT / f"{name}_s{seed}_{cond}.png"
                cfg = make_cfg(
                    sc,
                    prompt=sc["tgt_prompt"],
                    negative=sc["tgt_negative"],
                    seed=seed,
                    start_step=start_step,
                    output_path=str(image_path),
                    step_map=step_map,
                )
                plan = run(cfg, {"kind": "orig"})
                counts = plan.get("counts")
                image = Image.open(image_path).convert("RGB")
                diff = np.abs(np.asarray(image, dtype=np.float32) - input_rgb).mean(axis=2)
                record = dict(
                    scenario=name,
                    seed=seed,
                    condition=cond,
                    clip_edit=clip_edit(
                        clip_model, clip_processor, image, sc["caption_src"], sc["caption_tgt"]
                    ),
                    mae_all=float(diff.mean()),
                    mae_high=float(diff[high_px].mean()),
                    mae_low=float(diff[low_px].mean()),
                )
                if counts is not None:
                    depth = counts[0, 0].float()
                    record["mean_depth"] = float(depth.mean())
                    record["deep_frac"] = float((depth >= 0.9 * STEPS).float().mean())
                    record["corr"] = float(
                        np.corrcoef(depth.flatten().numpy(), signal.flatten().numpy())[0, 1]
                    )
                    if seed == seeds[0]:
                        first_seed_maps[cond] = counts.numpy()
                records.append(record)
                print(name, seed, cond, {k: round(v, 2) for k, v in record.items()
                                         if isinstance(v, float)}, flush=True)
            (OUT / "results.json").write_text(json.dumps(records, indent=1))

        montage(name, sc["input"], first_signal, first_seed_maps, condition_names(),
                OUT / f"{name}_montage.png")


if __name__ == "__main__":
    main()
