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

        self.flip_prob = 0
        self.aug_prob = 0
        self.shift_scale = (0.2, 0.4)
        self.right_prob = 0.5

        if self.split == "train":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_train.pkl")
            db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_train.pkl")
        elif self.split == "val":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_val.pkl")
        elif self.split == "trainval":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_trainval.pkl")
            db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_trainval.pkl")
        elif self.split == "test":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_test.pkl")
        else:
            raise ValueError("Invalid split!")

        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)
        self.num_samples = len(self.kitti_infos)
        
        if self.is_train:
            with open(db_info_path, 'rb') as f:
                db_infos = pickle.load(f)
            self.car_db_infos = db_infos["Car"]
            self.ped_db_infos = db_infos["Pedestrian"]
            self.cyc_db_infos = db_infos["Cyclist"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        info = self.kitti_infos[idx]
        img_path = os.path.join(self.kitti_root, "../" + info["img_path"])

        # use right image
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
            center_size = [center, size]
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return img, P2, center_size, image_idx
        
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

        # flip augmentation
        P2_backup = P2.copy()
        img_backup = img.copy()
        locations_backup = locations.copy()
        rotys_backup = rotys.copy()
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob) and (not use_right):
            img = img[:, ::-1, :]
            center[0] = size[0] - center[0] - 1
            P2[0, 2] = size[0] - P2[0, 2]  - 1
            flipped = True

        if (self.is_train) and flipped:
            temp = img.shape[1] - bboxes[:, 2] - 1
            bboxes[:, 2] = img.shape[1] - bboxes[:, 0] - 1
            bboxes[:, 0] = temp
            locations[:, 0] *= -1
            rotys *= -1

        # verify
        if (self.is_train) and flipped:
            if np.all(img_backup == img):
                img = img[:, ::-1, :]
                print("img flip failed ...")
            if np.all(P2_backup == P2):
                P2[0, 2] = size[0] - P2[0, 2]  - 1
                print("P2 flip failed ...")
            if np.all(locations_backup == locations):
                locations[:, 0] *= -1
                print("locations flip failed ...")
            if np.all(rotys_backup == rotys):
                rotys *= -1
                print("rotys flip failed ...")

        P = P3 if use_right else P2
        embedding_annos = []
        init_bboxes = []
        for i in range(len(names)):
            if names[i] not in self.classes:
                continue
            bbox = bboxes[i]
            locs = locations[i]
            rot_y = rotys[i]

            init_bboxes.append(bbox)
            ins_anno = {
                    "name": names[i],
                    "label": class_to_label[names[i]],
                    "bbox": bbox,
                    "alpha": alphas[i],
                    "dim": dimensions[i],
                    "loc": locs,
                    "roty": rot_y,
                    "P": P,
                    "difficulty": difficulty[i],
                    "truncated": truncated[i],
                    "occluded": occluded[i],
                    "flipped": flipped,
                    "score": scores[i]
                }
            embedding_annos.append(ins_anno)
        origin_num = len(embedding_annos)

        init_bboxes = np.array(init_bboxes)
        img_shape_key = f"{img.shape[0]}_{img.shape[1]}"

        if img_shape_key in self.car_db_infos.keys():
            car_db_infos_t = self.car_db_infos[img_shape_key]
            ins_ids = sample(range(len(car_db_infos_t)), min(16, self.max_objs - len(annos)))
            for i in ins_ids:
                ins = car_db_infos_t[i]
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
                '''
                if flipped:
                    P[0, 2] = size[0] - P[0, 2]  - 1
                    temp = img.shape[1] - box2d[2] - 1
                    box2d[2] = img.shape[1] - box2d[0] - 1
                    box2d[0] = temp
                '''

                if len(init_bboxes.shape) > 1:
                    ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                    if np.max(ious) > 0.0:
                        continue
                    init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                else:
                    init_bboxes = box2d[np.newaxis, ...].copy()            
                
                patch_img = cv2.imread(patch_img_path)
                ins_loc = ins["loc"]
                ins_roty = ins["roty"]
                '''
                if flipped:
                    patch_img = patch_img[:, ::-1, :]
                    ins_loc[0] *= -1
                    ins_roty *= -1
                '''
                img[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = patch_img
                ins_anno = {
                    "name": ins["name"],
                    "label": class_to_label[ins["name"]],
                    "bbox": box2d,
                    "alpha": ins["alpha"],
                    "dim": ins["dim"],
                    "loc": ins_loc,
                    "roty": ins_roty,
                    "P": P,
                    "difficulty": ins["difficulty"],
                    "truncated": ins["truncated"],
                    "occluded": ins["occluded"],
                    "flipped": flipped,
                    "score": ins["score"]
                }
                embedding_annos.append(ins_anno)  
        aug_num = len(embedding_annos) - origin_num
        # save_path = os.path.join("vis", str(image_idx)+".jpg")
        # self.visualization(img, embedding_annos, save_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)
        center_size = [center, size]

        return img, P2, embedding_annos, affine, flipped, center_size, image_idx

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
  