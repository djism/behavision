from behavision.video.reader import VideoReader


VIDEO_PATH = "/Users/dhananjaysharma/Desktop/videos/Trial1.mp4"


reader = VideoReader(VIDEO_PATH)

print("=== Video Metadata ===")
print("FPS:", reader.fps)
print("Frames:", reader.frame_count)
print(
    "Resolution:",
    reader.width,
    "x",
    reader.height
)
print(
    "Duration:",
    round(reader.duration, 2),
    "seconds"
)


print("\n=== Random Access Test ===")

frame = reader.get_frame(3000)

if frame is not None:
    print(
        "Frame:",
        frame.index,
        "Timestamp:",
        round(frame.timestamp, 2),
        "Shape:",
        frame.image.shape
    )


print("\n=== Stride Sampling Test ===")

reader.seek(0)

count = 0

for frame in reader.frames(stride=100):
    count += 1

print("Sampled frames:", count)


print("\n=== Time-Based Sampling Test ===")

reader.seek(0)

for frame in reader.frames_every(seconds=5):
    print(
        "Frame:",
        frame.index,
        "Timestamp:",
        round(frame.timestamp, 2)
    )


reader.release()