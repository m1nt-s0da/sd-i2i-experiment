from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from PIL import Image

from .config import ConfigI2I
from ._step_map import compute_step_counts
from tqdm import tqdm


def _save_latent_distance_plot(
    distances: list[list[float]],
    coordinates: list[tuple[int, int]],
    output_path: str,
    *,
    title: str = "Latent pixel distance between denoising steps",
    first_step: int = 1,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    for coordinate, pixel_distances in zip(coordinates, distances, strict=True):
        axis.plot(
            range(first_step, first_step + len(pixel_distances)),
            pixel_distances,
            label=f"latent ({coordinate[0]}, {coordinate[1]})",
        )
    axis.set(
        title=title,
        xlabel="Denoising step",
        ylabel="L2 distance",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _save_step_map_heatmap(
    step_counts: torch.Tensor,
    output_path: str,
    *,
    height: int,
    width: int,
    label: str = "i2i steps per latent pixel",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    heatmap = (
        torch.nn.functional.interpolate(
            step_counts[:1].float(),
            size=(height, width),
            mode="nearest",
        )[0, 0]
        .cpu()
        .numpy()
    )
    max_depth = max(1, int(heatmap.max()))

    figure, axis = plt.subplots()
    image = axis.imshow(heatmap, cmap="inferno", vmin=0, vmax=max_depth)
    axis.set_axis_off()
    figure.colorbar(
        image,
        ax=axis,
        label=label,
        ticks=MaxNLocator(integer=True, nbins=10),
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_image_generation(config: ConfigI2I) -> None:
    if not 0 <= config.start_step < config.steps:
        raise ValueError(
            f"start_step must be in [0, {config.steps}), got {config.start_step}."
        )
    if config.latent_pixel_count <= 0:
        raise ValueError(
            f"latent_pixel_count must be positive, got {config.latent_pixel_count}."
        )

    pipeline = config.pipeline.build_pipeline()
    if config.step_map is not None and config.pipeline.sampler not in (
        "euler",
        "ddim",
    ):
        raise ValueError("step_map supports only the euler and ddim samplers.")
    input_image = config.input.generate(pipeline=pipeline)

    input_tensor = pipeline.image_processor.preprocess(input_image).to(
        device=pipeline.device, dtype=pipeline.vae.dtype
    )
    controlnets = [
        (
            controlnet_config.build_controlnet(
                dtype=pipeline.unet.dtype,
                device=pipeline.device,
            ),
            controlnet_config.conditioning_scale,
            controlnet_config.create_condition(
                input_image,
                device=pipeline.device,
                dtype=pipeline.unet.dtype,
            ),
        )
        for controlnet_config in config.controlnet
    ]

    generator = (
        torch.Generator(device=pipeline.device).manual_seed(config.seed)
        if config.seed is not None
        else None
    )
    pipeline.scheduler.set_timesteps(config.steps, device=pipeline.device)
    timesteps = pipeline.scheduler.timesteps
    denoising_timesteps = timesteps[config.start_step :]
    if hasattr(pipeline.scheduler, "set_begin_index"):
        pipeline.scheduler.set_begin_index(config.start_step * pipeline.scheduler.order)
    latent_shape = (
        input_tensor.shape[0],
        pipeline.unet.config.in_channels,
        input_tensor.shape[-2] // pipeline.vae_scale_factor,
        input_tensor.shape[-1] // pipeline.vae_scale_factor,
    )
    input_latent = None
    noise = None
    if config.start_step == 0 and config.step_map is None:
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=pipeline.device,
            dtype=pipeline.vae.dtype,
        )
    else:
        with torch.no_grad():
            input_latent = pipeline.vae.encode(input_tensor).latent_dist.sample(
                generator=generator
            )
        input_latent = input_latent * pipeline.vae.config.scaling_factor
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=pipeline.device,
            dtype=pipeline.vae.dtype,
        )
        if config.start_step == 0:
            # 最深pixel（step 0開始）が真のノイズ（最大ノイズ量）から始まるようにする。
            latents = noise * pipeline.scheduler.init_noise_sigma
        else:
            latents = pipeline.scheduler.add_noise(
                input_latent, noise, denoising_timesteps[:1]
            )

    latent_height, latent_width = latents.shape[-2:]
    coordinate_generator = torch.Generator().manual_seed(
        0 if config.seed is None else config.seed
    )
    coordinate_indices = torch.randperm(
        latent_height * latent_width, generator=coordinate_generator
    )[: min(config.latent_pixel_count, latent_height * latent_width)]
    coordinates = [
        (int(index.item()) // latent_width, int(index.item()) % latent_width)
        for index in coordinate_indices
    ]
    distances = [[] for _ in coordinates]
    x0_distances = [[] for _ in coordinates]
    previous_x0 = None

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=config.prompt,
        device=pipeline.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=config.guidance_scale > 1.0,
        negative_prompt=config.negative_prompt,
    )
    do_classifier_free_guidance = config.guidance_scale > 1.0
    if not isinstance(prompt_embeds, torch.Tensor):
        raise RuntimeError("Expected prompt encoding to produce a tensor.")
    if do_classifier_free_guidance and not isinstance(
        negative_prompt_embeds, torch.Tensor
    ):
        raise RuntimeError(
            "Classifier-free guidance requires negative prompt embeddings."
        )
    prompt_embedding_tensor = cast(torch.Tensor, prompt_embeds)
    negative_prompt_embedding_tensor = cast(torch.Tensor, negative_prompt_embeds)
    encoder_hidden_states = prompt_embedding_tensor
    if do_classifier_free_guidance:
        encoder_hidden_states = torch.cat(
            [negative_prompt_embedding_tensor, prompt_embedding_tensor]
        )

    def predict_noise(model_input: torch.Tensor, timestep: torch.Tensor):
        if do_classifier_free_guidance:
            model_input = torch.cat([model_input] * 2)
        controlnet_residuals = None
        mid_block_residual = None
        if controlnets:
            outputs = [
                controlnet(
                    model_input,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_condition.repeat(
                        model_input.shape[0], 1, 1, 1
                    ),
                    conditioning_scale=conditioning_scale,
                    return_dict=False,
                )
                for controlnet, conditioning_scale, control_condition in controlnets
            ]
            controlnet_residuals = tuple(
                torch.stack([output[0][index] for output in outputs]).sum(dim=0)
                for index in range(len(outputs[0][0]))
            )
            mid_block_residual = torch.stack([output[1] for output in outputs]).sum(
                dim=0
            )
        noise_prediction = pipeline.unet(
            model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_residuals,
            mid_block_additional_residual=mid_block_residual,
            return_dict=False,
        )[0]
        if do_classifier_free_guidance:
            noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
            noise_prediction = noise_prediction_uncond + config.guidance_scale * (
                noise_prediction_text - noise_prediction_uncond
            )
        return noise_prediction

    step_counts = None
    with torch.no_grad():
        if config.step_map is not None:
            step_counts = compute_step_counts(
                latents,
                scheduler=pipeline.scheduler,
                start_step=config.start_step,
                config=config.step_map,
                predict_noise=predict_noise,
            )
        for step_index, timestep in enumerate(
            tqdm(denoising_timesteps), start=config.start_step
        ):
            if step_counts is not None:
                active = config.steps - step_counts <= step_index
                held = pipeline.scheduler.add_noise(
                    input_latent, noise, timestep.reshape(1)
                )
                latents = torch.where(active, latents, held.to(latents.dtype))
            previous_latents = latents
            latent_model_input = pipeline.scheduler.scale_model_input(latents, timestep)
            noise_prediction = predict_noise(latent_model_input, timestep)
            step_output = pipeline.scheduler.step(
                noise_prediction, timestep, latents, return_dict=True
            )
            current_x0 = getattr(step_output, "pred_original_sample", None)
            if current_x0 is None:
                # x0を公開しないスケジューラ（例: DPM-Solver）では、epsilon予測のVP式から算出する。
                alpha_prod = pipeline.scheduler.alphas_cumprod.to(
                    device=latents.device, dtype=latents.dtype
                )[int(timestep)]
                current_x0 = (
                    latents - (1 - alpha_prod).sqrt() * noise_prediction
                ) / alpha_prod.sqrt()
            latents = step_output.prev_sample
            if previous_x0 is not None:
                for x0_distances_for_pixel, (row, column) in zip(
                    x0_distances, coordinates, strict=True
                ):
                    x0_distances_for_pixel.append(
                        torch.linalg.vector_norm(
                            current_x0[0, :, row, column]
                            - previous_x0[0, :, row, column]
                        ).item()
                    )
            previous_x0 = current_x0
            for distances_for_pixel, (row, column) in zip(
                distances, coordinates, strict=True
            ):
                distances_for_pixel.append(
                    torch.linalg.vector_norm(
                        latents[0, :, row, column] - previous_latents[0, :, row, column]
                    ).item()
                )

        decoded_latents = latents / pipeline.vae.config.scaling_factor
        output_image = pipeline.vae.decode(decoded_latents, return_dict=False)[0]
        output_images = pipeline.image_processor.postprocess(
            output_image, output_type="pil"
        )
        if not isinstance(output_images, list) or not output_images:
            raise RuntimeError(
                "Expected the decoded output to be a non-empty image list."
            )
        output_image = output_images[0]
        if not isinstance(output_image, Image.Image):
            raise RuntimeError("Expected the decoded output to be a PIL image.")

    if (
        config.step_map is not None
        and config.step_map.heatmap_path
        and step_counts is not None
    ):
        _save_step_map_heatmap(
            step_counts,
            config.step_map.heatmap_path,
            height=input_image.height,
            width=input_image.width,
        )

    if config.latent_distance_plot_path:
        _save_latent_distance_plot(
            distances, coordinates, config.latent_distance_plot_path
        )
    if config.x0_distance_plot_path:
        _save_latent_distance_plot(
            x0_distances,
            coordinates,
            config.x0_distance_plot_path,
            title="Predicted x0 pixel distance between denoising steps",
            first_step=2,
        )
    if config.output_path:
        output = Path(config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output)


def main():
    from .config import load_config
    from argparse import ArgumentParser
    from ._main import run_image_generation
    import os

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="demo.toml")
    args = parser.parse_args()
    config_ext = os.path.splitext(args.config)
    config = load_config(args.config, config_ext[1][1:])
    run_image_generation(config.i2i)
