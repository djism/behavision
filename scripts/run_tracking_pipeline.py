from behavision.pipeline.tracking_pipeline import (
    TrackingPipeline,
)


VIDEO_PATH = "data/new_mouse_300frames1.mp4"
OUTPUT_DIR = "data/outputs/new_mouse_test"
CACHE_DIR = "data/cache/new_mouse_test"


pipeline = TrackingPipeline(
    prompt="black laboratory mouse",
    chunk_size=300,
)


pipeline.run(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    cache_dir=CACHE_DIR,
    overwrite_cache=False,
)