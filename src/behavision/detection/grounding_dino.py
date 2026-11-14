from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from behavision.detection.types import Detection


class GroundingDINODetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ):
        self.device = device or self._get_device()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(
            f"Loading Grounding DINO on device: "
            f"{self.device}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

        self.model.eval()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def detect(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> List[Detection]:

        pil_image = self._to_pil(image)

        text = prompt.strip().lower()

        # Grounding DINO generally works better with
        # labels separated / terminated by periods.
        if not text.endswith("."):
            text += "."

        inputs = self.processor(
            images=pil_image,
            text=text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[
                (
                    pil_image.height,
                    pil_image.width,
                )
            ],
        )[0]

        detections: List[Detection] = []

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["text_labels"]

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            x1, y1, x2, y2 = box.tolist()

            detection = Detection(
                class_name=str(label),
                confidence=float(score),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )

            detections.append(detection)

        return detections

    def _to_pil(
        self,
        image: np.ndarray,
    ) -> Image.Image:
        """
        VideoReader returns OpenCV BGR images.

        PIL / Grounding DINO expects RGB.
        """
        rgb_image = image[:, :, ::-1]

        return Image.fromarray(rgb_image)