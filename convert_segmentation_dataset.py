import os
from pathlib import Path
import shutil

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def main() -> None:
    data_path = "data_coco_format_20220724_154038"
    save_dir = data_path + "_np_mask"
    coco = COCO(data_path + "/result.json")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (img_id, data) in enumerate(tqdm(coco.imgs.items())):
        filename = data["file_name"]
        image_path = data_path + "/images/" + Path(filename).name
        h, w = data["height"], data["width"]
        annIds = coco.getAnnIds([img_id])
        anns = coco.loadAnns(annIds)
        anns_img = np.zeros((h, w))
        for ann in anns:
            anns_img += coco.annToMask(ann) > 0
        sample_save_dir = save_dir + f"/{Path(filename).stem}"
        if not os.path.exists(sample_save_dir):
            os.makedirs(sample_save_dir)
        np.save(sample_save_dir + "/mask.npy", anns_img)
        shutil.copy2(image_path, sample_save_dir)

        # try:
        assert np.sum(anns_img) > 0
        # except:
        #     print(i, filename)
        #     raise


if __name__ == "__main__":
    main()
