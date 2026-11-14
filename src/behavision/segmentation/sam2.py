from dataclasses import dataclass

import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class SegmentationResult:
    mask: np.ndarray
    score: float


class SAM2Segmenter:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        device: str | None = None,
    ):
        self.device = device or self._get_device()

        print(
            f"Loading SAM2 on device: {self.device}"
        )

        self.predictor = (
            SAM2ImagePredictor.from_pretrained(
                model_id,
                device=self.device,
            )
        )

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def segment_box(
        self,
        image: np.ndarray,
        box: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> SegmentationResult:

        # VideoReader gives BGR.
        # SAM2 expects RGB.
        rgb_image = image[:, :, ::-1].copy()

        self.predictor.set_image(
            rgb_image
        )

        box_array = np.array(
            [
                box[0],
                box[1],
                box[2],
                box[3],
            ],
            dtype=np.float32,
        )

        masks, scores, _ = (
            self.predictor.predict(
                box=box_array,
                multimask_output=True,
            )
        )

        best_index = int(
            np.argmax(scores)
        )

        best_mask = (
            masks[best_index]
            .astype(bool)
        )

        best_score = float(
            scores[best_index]
        )

        return SegmentationResult(
            mask=best_mask,
            score=best_score,
        )