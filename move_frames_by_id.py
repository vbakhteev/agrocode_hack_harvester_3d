import json
from pathlib import Path
import typing as tp

import cv2

from src.data_steam import DataStream


def main():
    with open("to_label.json") as f:
        video_to_ids: tp.Dict[str, tp.List[int]] = json.load(f)
    root_path = Path.cwd() / "frames"
    if not root_path.exists():
        root_path.mkdir()
    for video, ids in video_to_ids.items():
        istream = DataStream("data/" + video)
        for idx in ids:
            sample = istream[idx]
            save_path = f"frames/{video}_{idx}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(sample["color_frame"], cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
