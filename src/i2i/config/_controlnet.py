from dataclasses import dataclass
from typing import Literal
import os

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.controlnets.controlnet import ControlNetModel
from PIL import Image


@dataclass
class ConfigI2IControlNet:
    model: str
    conditioning_scale: float = 1.0

    def build_controlnet(
        self, *, dtype: torch.dtype, device: str | torch.device
    ) -> ControlNetModel:
        controlnet: ControlNetModel = ControlNetModel.from_pretrained(
            self.model, dtype=dtype
        )
        return ControlNetModel.to(controlnet, device=device)


@dataclass
class ConfigI2IControlNetDepth(ConfigI2IControlNet):
    type: Literal["depth"] = "depth"
    model: str = "lllyasviel/sd-controlnet-depth"
    condition_image: str | None = None
    save_image: str | None = None
    estimation_model: str = "Intel/dpt-hybrid-midas"

    def create_condition(
        self,
        image: Image.Image,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.condition_image is not None:
            with Image.open(self.condition_image) as source:
                condition_image = source.copy()
        else:
            from transformers import pipeline as transformers_pipeline

            depth_estimator = transformers_pipeline(
                "depth-estimation",
                model=self.estimation_model,
                device="cpu",
            )
            result = depth_estimator(image.convert("RGB"))
            if not isinstance(result, dict):
                raise RuntimeError("Expected depth estimation to return one result.")
            condition_image = result.get("depth")
            if not isinstance(condition_image, Image.Image):
                raise RuntimeError(
                    "Expected depth estimation to return a PIL depth map."
                )

            if self.save_image is not None:
                os.makedirs(os.path.dirname(self.save_image) or ".", exist_ok=True)
                condition_image.save(self.save_image)

        processor = VaeImageProcessor(
            vae_scale_factor=1,
            do_normalize=False,
            do_convert_rgb=True,
        )
        return processor.preprocess(
            condition_image, height=condition_image.height, width=condition_image.width
        ).to(device=device, dtype=dtype)


type ConfigI2IControlNets = ConfigI2IControlNetDepth
