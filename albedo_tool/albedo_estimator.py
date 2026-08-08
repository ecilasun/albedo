"""Inference engine for a pretrained intrinsic-decomposition model."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from albedo_tool.utils import (
    build_output_path,
    get_image_paths,
    image_to_tensor,
    load_image,
    save_image,
    tensor_to_image,
)
from models.intrinsic_decomposition import IntrinsicDecompositionNet

MODEL_REPOSITORY = "ssy1245/Intrinsic_Decomposition"
MODEL_FILENAME = "full_v4/model_final.pth"
MODEL_REVISION = "da2b229a626c617795cc25c34bdc5a8ac3813cb9"


class AlbedoEstimator:
    """Estimate reflectance and shading using a verified public checkpoint."""

    def __init__(
        self,
        device: str = "cpu",
    ):
        """Initialize tiled inference for the selected device."""
        self.device = torch.device(device)
        self.model: Optional[IntrinsicDecompositionNet] = None
        self.tile_size = 512
        self.tile_overlap = 128

    def load_model(self, checkpoint_path: Optional[str] = None) -> str:
        """Load a local checkpoint or download the default compatible checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=MODEL_FILENAME,
                revision=MODEL_REVISION,
            )
        elif not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        shading_weight = state_dict.get("shading_head.1.weight")
        if shading_weight is None:
            raise ValueError("Checkpoint has no shading_head.1.weight entry")

        model = IntrinsicDecompositionNet(
            color_shading=shading_weight.shape[0] == 3,
        )
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()
        return checkpoint_path

    def _tile_starts(self, length: int) -> List[int]:
        stride = self.tile_size - self.tile_overlap
        starts = list(range(0, length - self.tile_size + 1, stride))
        last_start = length - self.tile_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _window(self) -> torch.Tensor:
        coordinates = torch.linspace(-1, 1, self.tile_size, device=self.device)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        window = torch.exp(-(grid_x.square() + grid_y.square()) / (2 * 0.8**2))
        return window.unsqueeze(0).unsqueeze(0)

    def _infer(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() before estimating albedo.")

        _, _, height, width = image.shape
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32
        if pad_height or pad_width:
            image = functional.pad(
                image,
                (0, pad_width, 0, pad_height),
                mode="reflect",
            )

        _, _, padded_height, padded_width = image.shape
        if padded_height <= self.tile_size and padded_width <= self.tile_size:
            with torch.inference_mode():
                albedo, shading = self.model(image.to(self.device))
            return albedo[:, :, :height, :width], shading[:, :, :height, :width]

        albedo_map = torch.zeros(1, 3, padded_height, padded_width, device=self.device)
        shading_channels = 3 if self.model.color_shading else 1
        shading_map = torch.zeros(
            1,
            shading_channels,
            padded_height,
            padded_width,
            device=self.device,
        )
        weight_map = torch.zeros(1, 1, padded_height, padded_width, device=self.device)
        window = self._window()

        with torch.inference_mode():
            for y in self._tile_starts(padded_height):
                for x in self._tile_starts(padded_width):
                    tile = image[:, :, y : y + self.tile_size, x : x + self.tile_size]
                    albedo, shading = self.model(tile.to(self.device))
                    albedo_map[:, :, y : y + self.tile_size, x : x + self.tile_size] += albedo * window
                    shading_map[:, :, y : y + self.tile_size, x : x + self.tile_size] += shading * window
                    weight_map[:, :, y : y + self.tile_size, x : x + self.tile_size] += window

        albedo_map = albedo_map / weight_map.clamp_min(1e-8)
        shading_map = shading_map / weight_map.clamp_min(1e-8)
        return albedo_map[:, :, :height, :width], shading_map[:, :, :height, :width]

    def estimate_albedo(
        self,
        image: np.ndarray,
        resize: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return uint8 RGB albedo and uint8 shading images for an RGB image."""
        image_tensor = torch.from_numpy(image_to_tensor(image, resize=resize)).unsqueeze(0)
        albedo, shading = self._infer(image_tensor)
        albedo_image = tensor_to_image(albedo.squeeze(0).cpu().numpy())
        shading_array = shading.squeeze(0).cpu().numpy()
        if shading_array.shape[0] == 1:
            shading_array = np.repeat(shading_array, 3, axis=0)
        shading_image = tensor_to_image(shading_array)
        return albedo_image, shading_image

    def estimate_albedo_batch(
        self,
        input_path: str,
        output_dir: str,
        resize: Optional[int] = None,
        output_format: str = "png",
        overwrite: bool = False,
        verbose: bool = False,
    ) -> List[str]:
        """Estimate albedo for every supported image below an input directory."""
        if not os.path.isdir(input_path):
            raise NotADirectoryError(f"Input directory not found: {input_path}")

        image_paths = get_image_paths(input_path)
        if not image_paths:
            raise FileNotFoundError(f"No image files found in: {input_path}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_paths = []
        for image_path in tqdm(image_paths, desc="Processing"):
            output_path = build_output_path(image_path, output_dir, output_format)
            if os.path.exists(output_path) and not overwrite:
                if verbose:
                    print(f"Skipping (exists): {output_path}")
                continue

            image, _ = load_image(image_path)
            albedo, _ = self.estimate_albedo(image, resize=resize)
            save_image(albedo, output_path, output_format=output_format.upper())
            output_paths.append(output_path)
            if verbose:
                print(f"Saved: {output_path}")

        return output_paths
