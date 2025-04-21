from pathlib import Path
from pointcloudifier import PointCloudifier


def main() -> None:
    # Resolve the image path relative to this script.
    img_path = (Path(__file__).parent / "../data/miezekatze-512.png").resolve()

    pc = PointCloudifier()
    pc.cloudify_image(img_path, sample_rate=3, weights=(0.299, 0.587, 0.114))
    pc.save_json("miezekatze_cloud.json")
    pc.plot(point_size=1.0, title="Miezekatze point cloud")


if __name__ == "__main__":
    main()
