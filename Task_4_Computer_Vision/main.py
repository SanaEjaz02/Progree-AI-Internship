"""Run the Task 4 computer vision pipeline."""

from pathlib import Path

from pipeline import run_pipeline
from video_input.synthetic import generate_synthetic_video


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    source = ARTIFACTS / "synthetic_input.mp4"
    generate_synthetic_video(source)
    result = run_pipeline(source, ARTIFACTS)
    print(f"Source video: {source}")
    print(f"Annotated video: {result['video']}")
    print(f"Frame metrics: {result['metrics']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
