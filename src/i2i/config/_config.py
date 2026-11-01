from dataclasses import dataclass, field
from typing import Literal
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
from .._dtype import dtypes
from PIL import Image
import os
from ._controlnet import ConfigI2IControlNets


@dataclass
class ConfigI2IInputGenerate:
    type: Literal["generate"]
    prompt: str
    negative_prompt: str | None = None
    steps: int = 30
    guidance_scale: float = 8.0
    width: int = 512
    height: int = 512
    seed: int | None = None
    save_path: str | None = None

    def generate(self, *, pipeline: StableDiffusionPipeline, **kwargs) -> Image.Image:
        generator = (
            torch.Generator(device=pipeline.device).manual_seed(self.seed)
            if self.seed is not None
            else None
        )
        with torch.no_grad():
            result = pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                width=self.width,
                height=self.height,
                generator=generator,
                return_dict=True,
            )
        if not isinstance(result, StableDiffusionPipelineOutput):
            raise RuntimeError("Expected the pipeline to return an image result.")
        if not isinstance(result.images, list) or not result.images:
            raise RuntimeError("Expected the pipeline to return a non-empty image list.")
        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError("Expected the generated output to be a PIL image.")
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            image.save(self.save_path)
        return image


@dataclass
class ConfigI2IInputImage:
    type: Literal["image"]
    path: str

    def generate(self, **kwargs) -> Image.Image:
        return Image.open(self.path)


@dataclass
class ConfigI2IPipeline:
    architecture: Literal["sd15"]
    model: str
    dtype: str | None
    device: str | None
    vae: ConfigI2IVAE | None
    sampler: Literal["ddim", "euler"] = "euler"
    enable_attention_slicing: bool = False
    enable_model_cpu_offload: bool = False

    def build_pipeline(self) -> StableDiffusionPipeline:
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            dtype=dtypes[self.dtype] if self.dtype else None,
            safety_checker=None,
        )

        scheduler_config = pipeline.scheduler.config
        match self.sampler:
            case "ddim":
                pipeline.scheduler = DDIMScheduler.from_config(
                    scheduler_config, clip_sample=False
                )
            case "euler":
                pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    scheduler_config
                )

        if self.vae:
            pipeline.vae = self.vae.build_vae(pipeline.vae)

        if self.enable_attention_slicing:
            pipeline.enable_attention_slicing()

        if self.device:
            if self.enable_model_cpu_offload:
                pipeline.enable_model_cpu_offload(device=self.device)
            else:
                pipeline = pipeline.to(self.device)

        return pipeline


@dataclass
class ConfigI2IVAE:
    model: str | None = None
    enable_slicing: bool = False

    def build_vae(self, default: AutoencoderKL) -> AutoencoderKL:
        vae = default
        if self.model:
            vae = AutoencoderKL.from_pretrained(self.model)
        vae.enable_slicing()
        return vae


@dataclass
class ConfigI2IStepMap:
    # probe_stepsステップ試走し、x0の変化量が空間平均のthreshold倍を超えた回数に応じて
    # pixelごとのi2iステップ数（refine_steps〜残りステップ数）を決める。
    probe_steps: int = 4
    refine_steps: int = 1
    threshold: float = 1.0
    pool_kernel: int = 3
    heatmap_path: str | None = None


@dataclass
class ConfigI2I:
    input: ConfigI2IInputGenerate | ConfigI2IInputImage
    pipeline: ConfigI2IPipeline
    prompt: str
    controlnet: list[ConfigI2IControlNets] = field(default_factory=list)
    negative_prompt: str | None = None
    guidance_scale: float = 8.0
    steps: int = 30
    start_step: int = 10
    seed: int | None = None
    latent_pixel_count: int = 8
    latent_distance_plot_path: str | None = None
    x0_distance_plot_path: str | None = None
    output_path: str | None = ".var/i2i.png"
    step_map: ConfigI2IStepMap | None = None
