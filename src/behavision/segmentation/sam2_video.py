from dataclasses import dataclass

import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


@dataclass
class TrackedMask:
    frame_index: int
    object_id: int
    mask: np.ndarray


class SAM2VideoSegmenter:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        device: str | None = None,
    ):
        self.device = device or self._get_device()

        print(
            f"Loading SAM2 video predictor on device: "
            f"{self.device}"
        )

        self.predictor = (
            SAM2VideoPredictor.from_pretrained(
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

    def initialize(
        self,
        video_path: str,
    ):
        return self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

    def add_box(
        self,
        state,
        frame_index: int,
        box: tuple[
            float,
            float,
            float,
            float,
        ],
        object_id: int = 1,
    ):
        box_array = np.array(
            box,
            dtype=np.float32,
        )

        return (
            self.predictor
            .add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=object_id,
                box=box_array,
            )
        )

    def propagate(
        self,
        state,
        start_frame_index: int,
        max_frames: int | None = None,
    ):
        for (
            frame_index,
            object_ids,
            mask_logits,
        ) in self.predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=start_frame_index,
            max_frame_num_to_track=max_frames,
        ):

            masks = (
                mask_logits > 0.0
            ).cpu().numpy()

            for object_index, object_id in enumerate(
                object_ids
            ):
                mask = masks[
                    object_index,
                    0,
                ]

                yield TrackedMask(
                    frame_index=frame_index,
                    object_id=int(object_id),
                    mask=mask.astype(bool),
                )