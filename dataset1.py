import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i, j : the row and column based on cell unit
            # x_cell, y_cell : the row and column coordinates in cell
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x -j, self.S * y - i

            # the height and width based on cell unit
            width_cell, height_cell = (
                width * self.S, 
                height * self.S
            )

            # In one cell only one object
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates  # coordinates for that cell

                label_matrix[i, j, class_label] = 1  # one_hot encoding for class_label

        return image, label_matrix

# 118,287 5,000
# 123,287

# 117263 4,953
# 122,226



    # ================================= #
    #             for Yolo v3           #
    # ================================= #

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLOCustomDataset(Dataset):
    def __init__(
        self,
        csv_file, 
        img_dir, label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_threshold = 0.5
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()  # np.roll : [class, x, y, w, h] -> [x, y, w, h, clsss]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # 6 : [conf, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)         # compute IoU with all anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]  # check for each scales for bboxes


            for anchor_idx in anchor_indices:
                # print("anchor_idx:", anchor_idx.item(), "iou:", iou_anchors[anchor_idx], "anchor:", self.anchors[anchor_idx], "bbox:", box[2:4])
                scale_idx = anchor_idx // self.num_anchors_per_scale        # scale: 0, 1, 2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale   # anchor: 0, 1, 2
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x)  # cell index


                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if anchor_taken != 1 and not has_anchor[scale_idx]:
                    
                    # To assign prediction score for other cells that is covered by bbox
                    x1, y1 = x - width/2, y - height/2
                    x2, y2 = x + width/2, y + height/2
                    first_cell_x, first_cell_y = int(x1 * S), int(y1 * S)
                    last_cell_x, last_cell_y = int(x2 * S), int(y2 * S)
                    if first_cell_x == S:
                        first_cell_x -= 1
                    if first_cell_y == S:
                        first_cell_y -= 1
                    if last_cell_x == S:
                        last_cell_x -= 1
                    if last_cell_y == S:
                        last_cell_y -= 1

                    c2 = width ** 2 + height ** 2

                    x_cell, y_cell = S*x - j, S*y - i                       # x, y value in cell. values between [0, 1]
                    width_cell, height_cell = width * S, height * S         # w, h value for scale S
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    """for each cell bounded by bounding box"""
                    for jj in range(first_cell_x, last_cell_x + 1):
                        for ii in range(first_cell_y, last_cell_y + 1):
                            if targets[scale_idx][anchor_on_scale, ii, jj, 0] != 1:             # neighbor cell이 다른 bbox의 center가 아닐경우
                                cell_cx, cell_cy = (jj / S) + (1/(2*S)), (ii / S) + (1/(2*S))
                                distance = ((x - cell_cx) ** 2) + ((y - cell_cy) ** 2)
                                normalized_distance = distance / c2     # normalized by bbox
                                score = (normalized_distance - 1) ** 600
                                # normalized_distance = np.trunc(score*10) / 10 # 소수점 1자리 이하 삭제
                                # normalized_distance = distance / 1      # normalized by image size (=1)
                                if ii == i and jj == j:                                                       # assign bbox for center point
                                    # print("===========found center point!!")
                                    targets[scale_idx][anchor_on_scale, ii, jj, 0] = 1
                                    targets[scale_idx][anchor_on_scale, ii, jj, 5] = int(class_label)
                                    targets[scale_idx][anchor_on_scale, ii, jj, 1:5] = box_coordinates
                                else:                                                                           # neighbor cell이 다른 center보다 가까울경우
                                    if targets[scale_idx][anchor_on_scale, ii, jj, 0] < score:        # 32                                                                  # assign for around center point
                                        targets[scale_idx][anchor_on_scale, ii, jj, 0] = score
                                        targets[scale_idx][anchor_on_scale, ii, jj, 5] = int(class_label)
                                        # print(targets[scale_idx][anchor_on_scale, ii, jj, 0])



                    has_anchor[scale_idx] = True
                    """    """

                elif anchor_taken != 1 and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1       # not to punish for having large iou
                

        return image, tuple(targets)
        

    # ================================= #
    #             for Yolo v3           #
    # ================================= #

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file, 
        img_dir, label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_threshold = 0.5
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()  # np.roll : [class, x, y, w, h] -> [x, y, w, h, clsss]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # 6 : [conf, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)         # compute IoU with all anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]  # check for each scales for bboxes


            for anchor_idx in anchor_indices:
                # print("anchor_idx:", anchor_idx.item(), "iou:", iou_anchors[anchor_idx], "anchor:", self.anchors[anchor_idx], "bbox:", box[2:4])
                scale_idx = anchor_idx // self.num_anchors_per_scale        # scale: 0, 1, 2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale   # anchor: 0, 1, 2
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x)  # cell index


                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    """ 기존 """
                    # print("===========found center point!!")
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1        # Assign Objectness
                    x_cell, y_cell = S*x - j, S*y - i                       # x, y value in cell. values between [0, 1]
                    width_cell, height_cell = width * S, height * S         # w, h value for scale S
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    """    """

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1       # not to punish for having large iou


        # count = 0
        # for t in targets:
        #     print("scale=========", count)
        #     for a in range(t.shape[0]):  # anchor index
        #         print("anchor index========", a)
        #         # print(t[a,:,:,0].shape)
        #         # print(t[a,:,:,0])
        #         for b in range(t.shape[1]):
        #             for c in range(t.shape[2]):
        #                 if t[a, b, c, 0] == 1:
        #                     print(b, c, '/', t.shape[1], t.shape[2])
        #                     count += 1 
        # print("Total gt_bbox for all scales ========>>>>>>>:", count)
                

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    # dataset = YOLODataset(
    #     "COCO/train.csv",
    #     "COCO/images/images/",
    #     "COCO/labels/labels_new/",
    #     S=[13, 26, 52],
    #     anchors=anchors,
    #     transform=transform,
    # )
    # dataset = YOLODataset(
    #     "../dataset/pascalvoc/8examples.csv",
    #     "../dataset/pascalvoc/images/",
    #     "../dataset/pascalvoc/labels/",
    #     S=[13, 26, 52],
    #     anchors=anchors,
    #     transform=transform,
    # )
    dataset = YOLOCustomDataset(
        "../dataset/pascalvoc/100examples.csv",
        "../dataset/pascalvoc/images/",
        "../dataset/pascalvoc/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    index = 0
    for x, y in loader:
        # index += 1
        # if index == 6:
        # ============= #
        # x : img
        # y : tuple[(N, 3, 13, 13, 6), (N, 3, 26, 26, 6), (N, 3, 52, 52, 6)]
        # ============= #
        print("=========================================================")
        boxes = []

        for i in range(y[0].shape[1]):                  # scale for loop
            anchor = scaled_anchors[i]
            print("anchor shape:", anchor.shape)
            print("y[",i,"]", y[i].shape)               # y[0] = [N, 3, 13, 13, 6]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        # boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        boxes = nms(boxes, iou_threshold=1, threshold=0.8, box_format="midpoint")
        # print("boxes", boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        print("=========================================================")
        # break
        


if __name__ == "__main__":
    test()