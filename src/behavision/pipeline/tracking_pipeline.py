import csv
import os

from collections import deque

import cv2
import numpy as np
import torch


from behavision.video.reader import (
    VideoReader,
)

from behavision.video.frame_cache import (
    FrameCache,
)

from behavision.detection.grounding_dino import (
    GroundingDINODetector,
)

from behavision.detection.postprocess import (
    non_max_suppression,
    select_best_detection,
)

from behavision.segmentation.sam2_video import (
    SAM2VideoSegmenter,
)

from behavision.tracking.trajectory import (
    mask_centroid,
    mask_area,
    displacement,
    speed,
    area_ratio,
)

from behavision.behavior.head import (
    mask_principal_axis,
    compute_recent_motion,
    choose_front_endpoint,
)


class TrackingPipeline:

    def __init__(
        self,
        prompt: str = "black laboratory mouse",
        object_id: int = 1,
        jump_threshold_px: float = 80.0,
        area_ratio_low: float = 0.5,
        area_ratio_high: float = 2.0,
        chunk_size: int = 300,
        orientation_history: int = 8,
    ):

        self.prompt = prompt

        self.object_id = object_id

        self.jump_threshold_px = (
            jump_threshold_px
        )

        self.area_ratio_low = (
            area_ratio_low
        )

        self.area_ratio_high = (
            area_ratio_high
        )

        self.chunk_size = (
            chunk_size
        )

        self.orientation_history = (
            orientation_history
        )


        self.detector = (
            GroundingDINODetector()
        )

        self.segmenter = (
            SAM2VideoSegmenter()
        )


    # =================================================
    # Public entry point
    # =================================================

    def run(
        self,
        video_path: str,
        output_dir: str,
        cache_dir: str,
        overwrite_cache: bool = False,
    ):

        print(
            "=== Behavision Tracking Pipeline ==="
        )


        os.makedirs(
            output_dir,
            exist_ok=True,
        )


        metadata = (
            self._prepare_video(
                video_path=video_path,
                cache_dir=cache_dir,
                overwrite_cache=(
                    overwrite_cache
                ),
            )
        )


        initial_detection = (
            self._detect_initial_mouse(
                cache_dir=cache_dir,
            )
        )


        trajectory = (
            self._track_video(
                cache_dir=cache_dir,
                output_dir=output_dir,
                metadata=metadata,
                initial_detection=(
                    initial_detection
                ),
            )
        )


        self._save_trajectory(
            trajectory=trajectory,
            output_dir=output_dir,
        )


        print(
            "\n=== Tracking Complete ==="
        )


        return trajectory


    # =================================================
    # Video preparation
    # =================================================

    def _prepare_video(
        self,
        video_path: str,
        cache_dir: str,
        overwrite_cache: bool,
    ):

        print(
            "\n=== Step 1: Video Ingestion ==="
        )


        reader = VideoReader(
            video_path
        )


        metadata = {
            "fps":
                reader.fps,

            "frame_count":
                reader.frame_count,

            "width":
                reader.width,

            "height":
                reader.height,

            "duration":
                reader.duration,
        }


        print(
            "FPS:",
            metadata["fps"]
        )

        print(
            "Frames:",
            metadata["frame_count"]
        )

        print(
            "Resolution:",
            metadata["width"],
            "x",
            metadata["height"],
        )

        print(
            "Duration:",
            round(
                metadata["duration"],
                2,
            )
        )


        reader.release()


        print(
            "\n=== Step 2: Frame Cache ==="
        )


        cache = FrameCache(
            cache_dir
        )


        cache.build(
            video_path=video_path,
            overwrite=overwrite_cache,
        )


        return metadata


    # =================================================
    # Initial detection
    # =================================================

    def _detect_initial_mouse(
        self,
        cache_dir: str,
    ):

        print(
            "\n=== Step 3: Initial Detection ==="
        )


        frame_path = os.path.join(
            cache_dir,
            "000000.jpg",
        )


        image = cv2.imread(
            frame_path
        )


        if image is None:

            raise RuntimeError(
                f"Could not load "
                f"{frame_path}"
            )


        detections = (
            self.detector.detect(
                image=image,
                prompt=self.prompt,
            )
        )


        print(
            "Raw detections:",
            len(detections)
        )


        detections = (
            non_max_suppression(
                detections,
                iou_threshold=0.5,
            )
        )


        print(
            "After NMS:",
            len(detections)
        )


        best = (
            select_best_detection(
                detections
            )
        )


        if best is None:

            raise RuntimeError(
                "Could not detect mouse "
                "in first frame."
            )


        print(
            "Selected detection:",
            best
        )


        return best


    # =================================================
    # Chunked tracking
    # =================================================

    def _track_video(
        self,
        cache_dir: str,
        output_dir: str,
        metadata: dict,
        initial_detection,
    ):

        print(
            "\n=== Step 4: "
            "Chunked SAM2 Tracking ==="
        )


        fps = metadata["fps"]

        frame_count = (
            metadata["frame_count"]
        )

        width = (
            metadata["width"]
        )

        height = (
            metadata["height"]
        )


        tracked_video_path = (
            os.path.join(
                output_dir,
                "tracked_video.mp4",
            )
        )


        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )


        video_writer = (
            cv2.VideoWriter(
                tracked_video_path,
                fourcc,
                fps,
                (
                    width,
                    height,
                ),
            )
        )


        if not video_writer.isOpened():

            raise RuntimeError(
                "Could not create "
                "tracked video"
            )


        trajectory = []


        previous_centroid = None

        previous_area = None

        previous_front_point = None


        # ---------------------------------------------
        # Keep recent centroids.
        #
        # We use several frames instead of only one
        # frame of velocity because segmentation
        # jitter can otherwise reverse orientation.
        # ---------------------------------------------

        centroid_history = deque(
            maxlen=self.orientation_history
        )


        trajectory_points = []


        # num_chunks = (
        #     frame_count
        #     + self.chunk_size
        #     - 1
        # ) // self.chunk_size

        num_chunks = 1


        print(
            "Chunk size:",
            self.chunk_size,
        )

        print(
            "Total chunks:",
            num_chunks,
        )


        for chunk_index in range(
            num_chunks
        ):

            global_start = (
                chunk_index
                * self.chunk_size
            )


            global_end = min(
                global_start
                + self.chunk_size,
                frame_count,
            )


            print(
                f"\n--- Chunk "
                f"{chunk_index + 1}/"
                f"{num_chunks}"
                f" | Frames "
                f"{global_start}-"
                f"{global_end - 1}"
                f" ---"
            )


            chunk_dir = (
                os.path.join(
                    output_dir,
                    "_chunks",
                    f"chunk_"
                    f"{chunk_index:04d}",
                )
            )


            os.makedirs(
                chunk_dir,
                exist_ok=True,
            )


            # =========================================
            # Build SAM2 chunk directory
            # =========================================

            local_index = 0


            for global_frame_index in range(
                global_start,
                global_end,
            ):

                source_path = (
                    os.path.join(
                        cache_dir,
                        f"{global_frame_index:06d}.jpg",
                    )
                )


                destination_path = (
                    os.path.join(
                        chunk_dir,
                        f"{local_index:06d}.jpg",
                    )
                )


                image = cv2.imread(
                    source_path
                )


                if image is None:

                    raise RuntimeError(
                        "Could not load "
                        "cached frame "
                        f"{global_frame_index}"
                    )


                success = (
                    cv2.imwrite(
                        destination_path,
                        image,
                    )
                )


                if not success:

                    raise RuntimeError(
                        "Could not save "
                        "chunk frame "
                        f"{destination_path}"
                    )


                local_index += 1


            # =========================================
            # Detection at chunk start
            # =========================================

            first_chunk_frame_path = (
                os.path.join(
                    chunk_dir,
                    "000000.jpg",
                )
            )


            first_chunk_frame = (
                cv2.imread(
                    first_chunk_frame_path
                )
            )


            if first_chunk_frame is None:

                raise RuntimeError(
                    "Could not load "
                    "first chunk frame"
                )


            # First chunk already has a detection.
            if chunk_index == 0:

                best = (
                    initial_detection
                )

            else:

                detections = (
                    self.detector.detect(
                        image=first_chunk_frame,
                        prompt=self.prompt,
                    )
                )


                detections = (
                    non_max_suppression(
                        detections,
                        iou_threshold=0.5,
                    )
                )


                best = (
                    select_best_detection(
                        detections
                    )
                )


            if best is None:

                raise RuntimeError(
                    "No mouse detection "
                    f"at frame "
                    f"{global_start}"
                )


            print(
                "Initialization detection:",
                best,
            )


            box = (
                best.x1,
                best.y1,
                best.x2,
                best.y2,
            )


            # =========================================
            # Initialize SAM2
            # =========================================

            state = (
                self.segmenter.initialize(
                    chunk_dir
                )
            )


            self.segmenter.add_box(
                state=state,
                frame_index=0,
                box=box,
                object_id=self.object_id,
            )


            # =========================================
            # Propagate masks
            # =========================================

            for tracked in (
                self.segmenter.propagate(
                    state=state,
                    start_frame_index=0,
                    max_frames=(
                        global_end
                        - global_start
                    ),
                )
            ):

                local_frame_index = (
                    tracked.frame_index
                )


                global_frame_index = (
                    global_start
                    + local_frame_index
                )


                frame_path = (
                    os.path.join(
                        cache_dir,
                        f"{global_frame_index:06d}.jpg",
                    )
                )


                image = cv2.imread(
                    frame_path
                )


                if image is None:

                    raise RuntimeError(
                        "Could not load frame "
                        f"{global_frame_index}"
                    )


                # =====================================
                # Mouse centroid
                # =====================================

                centroid = (
                    mask_centroid(
                        tracked.mask
                    )
                )


                if centroid is None:

                    video_writer.write(
                        image
                    )

                    continue


                cx, cy = centroid


                centroid_history.append(
                    centroid
                )


                # =====================================
                # Head / front estimation
                # =====================================

                head_geometry = (
                    mask_principal_axis(
                        tracked.mask
                    )
                )


                front_point = None


                if head_geometry is not None:

                    motion_direction = (
                        compute_recent_motion(
                            centroid_history,
                            min_displacement_px=2.0,
                        )
                    )


                    front_point = (
                        choose_front_endpoint(
                            endpoint_a=(
                                head_geometry[
                                    "endpoint_a"
                                ]
                            ),

                            endpoint_b=(
                                head_geometry[
                                    "endpoint_b"
                                ]
                            ),

                            centroid=centroid,

                            motion_direction=(
                                motion_direction
                            ),

                            previous_front=(
                                previous_front_point
                            ),
                        )
                    )


                # -------------------------------------
                # Store front point
                # -------------------------------------

                front_x = None
                front_y = None


                if front_point is not None:

                    front_x = float(
                        front_point[0]
                    )

                    front_y = float(
                        front_point[1]
                    )


                    previous_front_point = (
                        front_point
                    )


                # =====================================
                # Standard trajectory features
                # =====================================

                current_area = (
                    mask_area(
                        tracked.mask
                    )
                )


                frame_displacement = (
                    displacement(
                        previous_centroid,
                        centroid,
                    )
                )


                delta_time = (
                    1.0 / fps
                )


                speed_px_s = (
                    speed(
                        frame_displacement,
                        delta_time,
                    )
                )


                current_area_ratio = (
                    area_ratio(
                        previous_area,
                        current_area,
                    )
                )


                jump_flag = (
                    frame_displacement
                    is not None
                    and
                    frame_displacement
                    > self.jump_threshold_px
                )


                area_flag = False


                if (
                    current_area_ratio
                    is not None
                ):

                    area_flag = (
                        current_area_ratio
                        < self.area_ratio_low
                        or
                        current_area_ratio
                        > self.area_ratio_high
                    )


                timestamp = (
                    global_frame_index
                    / fps
                )


                # =====================================
                # Save trajectory row
                # =====================================

                trajectory.append(
                    {
                        "frame":
                            global_frame_index,

                        "timestamp":
                            timestamp,

                        "x":
                            cx,

                        "y":
                            cy,

                        "front_x":
                            front_x,

                        "front_y":
                            front_y,

                        "mask_area":
                            current_area,

                        "displacement":
                            frame_displacement,

                        "speed_px_s":
                            speed_px_s,

                        "area_ratio":
                            current_area_ratio,

                        "jump_flag":
                            jump_flag,

                        "area_flag":
                            area_flag,
                    }
                )


                trajectory_points.append(
                    (
                        int(cx),
                        int(cy),
                    )
                )


                # =====================================
                # Visualization
                # =====================================

                output = (
                    self._draw_tracking_overlay(
                        image=image,

                        mask=tracked.mask,

                        centroid=centroid,

                        front_point=(
                            front_point
                        ),

                        trajectory_points=(
                            trajectory_points
                        ),

                        frame_index=(
                            global_frame_index
                        ),

                        timestamp=(
                            timestamp
                        ),

                        speed_px_s=(
                            speed_px_s
                        ),

                        jump_flag=(
                            jump_flag
                        ),

                        area_flag=(
                            area_flag
                        ),
                    )
                )


                video_writer.write(
                    output
                )


                if (
                    global_frame_index
                    % 100
                    == 0
                ):

                    print(
                        f"Frame "
                        f"{global_frame_index}/"
                        f"{frame_count}"
                        f" | "
                        f"{timestamp:.1f}s"
                        f" | "
                        f"centroid="
                        f"({cx:.1f}, "
                        f"{cy:.1f})"
                        f" | "
                        f"front="
                        f"{front_point}"
                    )


                previous_centroid = (
                    centroid
                )


                previous_area = (
                    current_area
                )


            # =========================================
            # Release SAM2 state
            # =========================================

            del state


            if (
                torch.backends.mps.is_available()
            ):

                torch.mps.empty_cache()


        video_writer.release()


        print(
            "\nTracked video:",
            tracked_video_path,
        )


        return trajectory


    # =================================================
    # Visualization
    # =================================================

    def _draw_tracking_overlay(
        self,
        image,
        mask,
        centroid,
        front_point,
        trajectory_points,
        frame_index,
        timestamp,
        speed_px_s,
        jump_flag,
        area_flag,
    ):

        output = (
            image.copy()
        )


        # =============================================
        # Green SAM2 mask
        # =============================================

        overlay = (
            output.copy()
        )


        overlay[
            mask
        ] = (
            0,
            255,
            0,
        )


        output = (
            cv2.addWeighted(
                overlay,
                0.35,
                output,
                0.65,
                0,
            )
        )


        # =============================================
        # Centroid
        # =============================================

        cx = int(
            centroid[0]
        )

        cy = int(
            centroid[1]
        )


        cv2.circle(
            output,
            (cx, cy),
            6,
            (0, 0, 255),
            -1,
        )


        # =============================================
        # Estimated front / head
        # =============================================

        if front_point is not None:

            fx = int(
                front_point[0]
            )

            fy = int(
                front_point[1]
            )


            # Yellow head/front point
            cv2.circle(
                output,
                (fx, fy),
                8,
                (0, 255, 255),
                -1,
            )


            # Orientation vector
            cv2.line(
                output,
                (cx, cy),
                (fx, fy),
                (0, 255, 255),
                3,
            )


            cv2.putText(
                output,
                "FRONT",
                (
                    fx + 8,
                    fy - 8,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )


        # =============================================
        # Recent trajectory
        # =============================================

        recent_points = (
            trajectory_points[
                -150:
            ]
        )


        for i in range(
            1,
            len(recent_points),
        ):

            cv2.line(
                output,
                recent_points[i - 1],
                recent_points[i],
                (255, 0, 0),
                2,
            )


        # =============================================
        # Information
        # =============================================

        cv2.putText(
            output,
            f"Frame: {frame_index}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


        cv2.putText(
            output,
            f"Time: {timestamp:.2f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


        if speed_px_s is not None:

            cv2.putText(
                output,
                f"Speed: "
                f"{speed_px_s:.1f} px/s",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )


        # =============================================
        # Tracking warnings
        # =============================================

        if (
            jump_flag
            or area_flag
        ):

            cv2.putText(
                output,
                "TRACK WARNING",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )


        return output


    # =================================================
    # Save trajectory
    # =================================================

    def _save_trajectory(
        self,
        trajectory,
        output_dir,
    ):

        output_path = (
            os.path.join(
                output_dir,
                "trajectory.csv",
            )
        )


        fieldnames = [
            "frame",
            "timestamp",
            "x",
            "y",

            "front_x",
            "front_y",

            "mask_area",
            "displacement",
            "speed_px_s",
            "area_ratio",
            "jump_flag",
            "area_flag",
        ]


        with open(
            output_path,
            "w",
            newline="",
        ) as file:

            writer = (
                csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
            )


            writer.writeheader()


            writer.writerows(
                trajectory
            )


        print(
            "Trajectory:",
            output_path,
        )