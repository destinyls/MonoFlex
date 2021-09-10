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

        self.aug_prob = 0.0
        self.shift_scale = (0.2, 0.4)
        self.right_prob = 0.5
        self.bcp_prob = 0.5

        if self.split == "train":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_train.pkl")
            db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_test_48666.pkl")
        elif self.split == "val":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_val.pkl")
        elif self.split == "trainval":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_trainval.pkl")
            db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_test.pkl")
        elif self.split == "test":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_test_7518.pkl")
        else:
            raise ValueError("Invalid split!")

        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)
        self.num_samples = len(self.kitti_infos)

        if self.is_train:
            with open(db_info_path, 'rb') as f:
                self.db_infos = pickle.load(f)

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
        init_bboxes = []
        P = P3 if use_right else P2
        for i in range(len(names)):
            init_bboxes.append(bboxes[i])
            if names[i] not in self.classes:
                continue
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
        init_bboxes = np.array(init_bboxes)

        use_bcp = False
        if use_right or random.random() < self.bcp_prob:
            use_bcp = True

        if use_bcp:
            aug_class = "Car"
            img_shape_key = f"{img.shape[0]}_{img.shape[1]}"
            ori_annos_num = len(embedding_annos)
            if img_shape_key in self.db_infos[aug_class].keys():
                class_db_infos = self.db_infos[aug_class][img_shape_key]
                ins_ids = sample(range(len(class_db_infos)), min(16, self.max_objs - len(annos)))
                for ins_id in ins_ids:
                    ins = class_db_infos[ins_id]
                    patch_img_path = os.path.join(self.kitti_root, "../" + ins["path"])
                    if use_right:
                        box2d = ins["bbox_r"]
                        P = ins["P3"]
                        patch_img_path = patch_img_path.replace("image_2", "image_3")
                    else:
                        box2d = ins["bbox_l"]
                        P = ins["P2"]
                    
                    if ins['difficulty'] > 0:
                        continue
                    if ins['score'] < 0.75:
                        continue
                    if len(init_bboxes.shape) > 1:
                        ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                        if np.max(ious) > 0.0:
                            continue
                        init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                    else:
                        init_bboxes = box2d[np.newaxis, ...].copy()
                    patch_img = cv2.imread(patch_img_path)
                    img[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = patch_img
                    ins_anno = {
                        "name": ins["name"],
                        "label": class_to_label[ins["name"]],
                        "bbox": box2d,
                        "alpha": ins["alpha"],
                        "dim": ins["dim"],
                        "loc": ins["loc"],
                        "roty": ins["roty"],
                        "P": P,
                        "difficulty": ins["difficulty"],
                        "truncated": ins["truncated"],
                        "occluded": ins["occluded"],
                        "flipped": False,
                        "score": ins["score"]
                    }
                    embedding_annos.append(ins_anno) 
            aug_annos_num = len(embedding_annos)
            if ori_annos_num == aug_annos_num:
                use_bcp = False

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img, P, use_right, use_bcp, embedding_annos, image_idx

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