import math
from pathlib import Path
from pointcloudifier import PointCloudifier, quantised_values_rounded, quantised_values_exact


def main() -> None:
    # Resolve the image path relative to this script.
    img_path = (Path(__file__).parent / "../data/miezekatze-512.png").resolve()

    pc = PointCloudifier()
    pc.cloudify_image(img_path, sample_rate=3, weights=(0.299, 0.587, 0.114))
    pc.quantise_values(quantised_values_exact(-1.0, 1.0, 5))
    pc.drop_random(fraction=0.5, seed=42)
    pc.plot(point_size=1.0, title="Miezekatze Point Cloud")


if __name__ == "__main__":
    main()
