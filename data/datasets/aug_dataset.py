import cv2
import os
import pickle
import random
import numpy as np

from random import sample
from PIL import Image
from data.datasets import kitti_common as kitti

class AUGDataset():
    def __init__(self, cfg, kitti_root, is_train=True, split="train"):
        super(AUGDataset, self).__init__()
        self.kitti_root = kitti_root
        self.split = split
        self.is_train = is_train
        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.aug_prob = 0
        self.shift_scale = (0.2, 0.4)
        self.right_prob = 0
        if self.split == "train":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_train.pkl")
        elif self.split == "val":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_val.pkl")
        elif self.split == "trainval":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_trainval.pkl")
        elif self.split == "test":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_test.pkl")
        else:
            raise ValueError("Invalid split!")

        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)
        self.num_samples = len(self.kitti_infos)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        info = self.kitti_infos[idx]
        img_path = os.path.join(self.kitti_root, "../" + info["img_path"])

        use_right = False
        if self.is_train and random.random() < self.right_prob:
            use_right = True
            img_path = img_path.replace("image_2", "image_3")
        img = cv2.imread(img_path)
        image_idx = info["image_idx"]
        P2 = info["calib/P2"]
        P3 = info["calib/P3"]
        img_size = [img.shape[1], img.shape[0]]
        center = np.array([i / 2 for i in img_size], dtype=np.float32)
        size = np.array([i for i in img_size], dtype=np.float32)
        class_to_label = kitti.get_class_to_label_map()

        if not self.is_train:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return img, P2, image_idx

        annos = info["annos"]
        names = annos["name"] 
        bboxes = annos["bbox"]
        alphas = annos["alpha"]
        dimensions = annos["dimensions"]
        locations = annos["location"]
        rotys = annos["rotation_y"]
        difficulty = annos["difficulty"]
        truncated = annos["truncated"]
        occluded = annos["occluded"]
        scores = annos["score"]
        embedding_annos = []
        P = P3 if use_right else P2
        for i in range(len(names)):
            ins_anno = {
                    "name": names[i],
                    "label": class_to_label[names[i]],
                    "bbox": bboxes[i],
                    "alpha": alphas[i],
                    "dim": dimensions[i],
                    "loc": locations[i],
                    "roty": rotys[i],
                    "P": P,
                    "difficulty": difficulty[i],
                    "truncated": truncated[i],
                    "occluded": occluded[i],
                    "flipped": False,
                    "score": scores[i]
                }
            embedding_annos.append(ins_anno)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img, use_right, P, embedding_annos, image_idx

    def visualization(self, img, annos, save_path):
        image = img.copy()
        for anno in annos:
            dim = anno["dim"]
            loc = anno["loc"]
            roty = anno["roty"]
            bbox = anno["bbox"]
            box3d = kitti.compute_box_3d_image(anno["P"], roty, dim, loc)
            image = kitti.draw_box_3d(image, box3d)
        cv2.imwrite(save_path, image)
  