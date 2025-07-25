import helper
from pathlib import Path
from pointcloudifier import PointCloudifier


def main() -> None:
    # Resolve the image path relative to this script.
    img_path = (Path(__file__).parent / "../data/two-blobs-128.png").resolve()

    pc = PointCloudifier()
    pc.cloudify_image(img_path, sample_rate=1)
    pc.quantise_values(helper.quantised_values_exact(-1.0, 1.0, 3))
    pc.drop_below(-0.5)
    pc.drop_random(fraction=0.5, seed=42)
    pc.shake(0.2, 0.2, 0.0)
    pc.plot(point_size=1.0, title="Point Cloud", cmap="coolwarm")


if __name__ == "__main__":
    main()
