import torch
import torch.optim as optim

import config
from model import Yolov3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,get_evaluation_bboxes_accuracy,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,get_loaders_custom,
    plot_couple_examples,
    non_max_suppression as nms,
    plot_image
    )
from loss import YoloLoss_v3

def main():
    model = Yolov3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # ================ #
    # train: 16552
    # test: 4952
    # _, test_loader, _ = get_loaders_custom(
    #     train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv", batch_size=None
    # )
    # _, test_loader, _ = get_loaders(
    #     train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv", batch_size=1
    # )
    _, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/test.csv", batch_size=None
    )



    # CHECKPOINT_FILE = "plain_checkpoint100.pth"
    # CHECKPOINT_FILE = "custom_checkpoint100.pth"
    CHECKPOINT_FILE = "plain_checkpoint43223.pth"
    if config.LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    model.eval()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # [3] => [3,1] => [3,1,1] => [3,3,2]
    ).to(config.DEVICE)

    CONF_THRESHOLD = 0.2 # config.CONF_THRESHOLD  # 0.9
    for CONF_THRESHOLD in [0.2, 0.4, 0.6, 0.8, 0.9]:
        # Non Maximum Suppression                                                
        pred_boxes, true_boxes,_,_,_ = get_evaluation_bboxes_accuracy(
            test_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=CONF_THRESHOLD,
        )
        # Mean Average Precision
        print("pred box:", len(pred_boxes))
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")

    # index = 0
    # for x, y in test_loader:
    #     all_pred_boxes = []
    #     all_true_boxes = []
    #     # index += 1
    #     # if index == 6:
    #     # ============= #
    #     # x : img
    #     # y : tuple[(N, 3, 13, 13, 6), (N, 3, 26, 26, 6), (N, 3, 52, 52, 6)]
    #     # ============= #
    #     x = x.to(config.DEVICE)
    #     preds = model(x)
    #     print("=========================================================")
    #     # boxes = []

    #     batch_size = x.shape[0]
    #     bboxes = [[] for _ in range(batch_size)]
    #     for i in range(y[0].shape[1]):                  # scale for loop
    #         S = preds[i].shape[2]
    #         anchor = torch.tensor([*config.ANCHORS[i]]).to(config.DEVICE) * S
    #         print("anchor shape:", anchor.shape)
    #         print("preds[",i,"]", preds[i].shape)               # y[0] = [N, 3, 13, 13, 6]
    #         print("gt:",)
    #         boxes_scale_i = cells_to_bboxes(
    #             preds[i], is_preds=True, S=preds[i].shape[2], anchors=anchor
    #         )
    #         for idx, (box) in enumerate(boxes_scale_i):
    #             bboxes[idx] += box
    #     # print(len(bboxes))
    #     # # we just want one bbox for each label, not one for each scale
    #     true_bboxes = cells_to_bboxes(
    #         y[2], anchor, S=S, is_preds=False
    #     )
        

    #     for idx in range(batch_size):
    #         nms_boxes = nms(
    #             bboxes[idx],
    #             iou_threshold=config.NMS_IOU_THRESH,
    #             threshold=CONF_THRESHOLD,
    #             box_format="midpoint",
    #         )

    #         for nms_box in nms_boxes:
    #             all_pred_boxes.append(nms_box)
            
    #         for box in true_bboxes[idx]:
    #             if box[1] > CONF_THRESHOLD:
    #                 all_true_boxes.append(box)

    #     # nms_boxes = nms(boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=0.4, box_format="midpoint")
    #     # boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
    #     print("nms box", len(all_pred_boxes))
    #     print("gt boxes:", len(all_true_boxes))
    #     # print("boxes", boxes)
    #     plot_image(x[0].permute(1, 2, 0).to("cpu"), all_pred_boxes)
    #     print("=========================================================")
    #     # break


if __name__ == "__main__":
    main()