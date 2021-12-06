import config
import torch
import torch.optim as optim

import wandb
import pdb

from model import Yolov3
from tqdm import tqdm
from utils import (
    get_evaluation_bboxes_accuracy,
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,get_loaders_custom,
    plot_couple_examples
)
from loss import YoloLoss_v3, YoloLoss_v3_custom
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses, box_losses, obj_losses, noobj_losses, class_losses = [], [], [], [], []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            # loss = (
            #     loss_fn(out[0], y0, scaled_anchors[0])
            #     + loss_fn(out[1], y1, scaled_anchors[1])
            #     + loss_fn(out[2], y2, scaled_anchors[2])
            # )
            s1 = loss_fn(out[0], y0, scaled_anchors[0])
            s2 = loss_fn(out[1], y1, scaled_anchors[1])
            s3 = loss_fn(out[2], y2, scaled_anchors[2])

            box_loss = s1[0] + s2[0] + s3[0]
            obj_loss = s1[1] + s2[1] + s3[1]
            noobj_loss = s1[2] + s2[2] + s3[2]
            class_loss = s1[3] + s2[3] + s3[3]
            # class_loss = s1[2] + s2[2] + s3[2]

            loss = box_loss + obj_loss + noobj_loss + class_loss
            # loss = box_loss + obj_loss + class_loss

        losses.append(loss.item())
        box_losses.append(box_loss.item())
        obj_losses.append(obj_loss.item())
        noobj_losses.append(noobj_loss)
        class_losses.append(class_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    mean_box_loss = sum(box_losses) / len(box_losses)
    mean_obj_loss = sum(obj_losses) / len(obj_losses)
    mean_noobj_loss = sum(noobj_losses) / len(noobj_losses)
    mean_class_loss = sum(class_losses) / len(class_losses)


    return mean_loss, mean_box_loss, mean_obj_loss, mean_noobj_loss, mean_class_loss
    # return mean_loss, mean_box_loss, mean_obj_loss, mean_class_loss



def main():
    model = Yolov3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    wandb.watch(model)

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )


    # loss_fn = YoloLoss_v3()
    loss_fn = YoloLoss_v3_custom()
    scaler = torch.cuda.amp.GradScaler()

    # =======  Load plain dataset  ======= #
    # train_loader, test_loader, train_eval_loader = get_loaders(
    #     train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv", batch_size=None
    # )

    # =======  Load custom dataset  ======= #
    train_loader, test_loader, train_eval_loader = get_loaders_custom(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv", batch_size=None
    )
    # train_loader, test_loader, train_eval_loader = get_loaders_custom(
    #     train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv", batch_size=None
    # )


    # CHECKPOINT_FILE = "plain_checkpoint.pth"
    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    #     )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # [3] => [3,1] => [3,1,1] => [3,3,2]
    ).to(config.DEVICE)

    best_map = 0
    for epoch in range(config.NUM_EPOCHS):
        print("=====epochs:",epoch,"======")
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        # train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)


        # mean_loss, mean_box_loss, mean_obj_loss, mean_noobj_loss, mean_class_loss \
        #     = train_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)  # for testing train with 100 examples without augmentations
        mean_loss, mean_box_loss, mean_obj_loss, mean_noobj_loss, mean_class_loss \
            = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)  # for training with augmentation

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")        # validation
        if epoch > 0 and epoch % 10 == 0:
        # if epoch % 1 == 0:
            # print("================= On Train loader:")
            # train_class_acc, train_noobj_acc, train_obj_acc = check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

            print("================= On Test loader:")
            # confscore 0.2 기본
            print("=============Conf Threshold 0.2")
            # Check Accuracy
            # class_acc, noobj_acc, obj_acc = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # # Non Maximum Suppression                                                
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=config.CONF_THRESHOLD,
            # )                                              
            pred_boxes, true_boxes, class_acc, noobj_acc, obj_acc = get_evaluation_bboxes_accuracy(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
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
            print(f"MAP2: {mapval.item()}")


            # confscore 0.4
            print("=============Conf Threshold 0.4")
            CONF_THRESHOLD = 0.4
            # # Check Accuracy
            # _, _, obj_acc5 = check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            #  # Non Maximum Suppression                                                
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=CONF_THRESHOLD,
            # )                                               
            pred_boxes, true_boxes, _, _, obj_acc4 = get_evaluation_bboxes_accuracy(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=CONF_THRESHOLD,
            )
            # Mean Average Precision
            print("pred box:", len(pred_boxes))
            mapval_4 = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP4: {mapval_4.item()}")


            # confscore 0.6
            print("=============Conf Threshold 0.6")
            CONF_THRESHOLD = 0.6
            # # Check Accuracy
            # _, _, obj_acc6 = check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            # # Non Maximum Suppression                                                 
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=CONF_THRESHOLD,
            # )                                               
            pred_boxes, true_boxes, _, _, obj_acc6 = get_evaluation_bboxes_accuracy(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=CONF_THRESHOLD,
            )
            # Mean Average Precision
            print("pred box:", len(pred_boxes))
            mapval_6 = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP6: {mapval_6.item()}")


            # # confscore 0.7
            # print("=============Conf Threshold 0.7")
            # CONF_THRESHOLD = 0.7
            # # # Check Accuracy
            # # _, _, obj_acc7 = check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            # # # Non Maximum Suppression                                                 
            # # pred_boxes, true_boxes = get_evaluation_bboxes(
            # #     test_loader,
            # #     model,
            # #     iou_threshold=config.NMS_IOU_THRESH,
            # #     anchors=config.ANCHORS,
            # #     threshold=CONF_THRESHOLD,
            # # )                                               
            # pred_boxes, true_boxes, _, _, obj_acc7 = get_evaluation_bboxes_accuracy(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=CONF_THRESHOLD,
            # )
            # # Mean Average Precision
            # print("pred box:", len(pred_boxes))
            # mapval_7 = mean_average_precision(
            #     pred_boxes,
            #     true_boxes,
            #     iou_threshold=config.MAP_IOU_THRESH,
            #     box_format="midpoint",
            #     num_classes=config.NUM_CLASSES,
            # )
            # print(f"MAP7: {mapval_7.item()}")


            # confscore 0.8
            print("=============Conf Threshold 0.8")
            CONF_THRESHOLD = 0.8
            # Check Accuracy
            # _, _, obj_acc8 = check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            # # Non Maximum Suppression                                                 
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=CONF_THRESHOLD,
            # )                                               
            pred_boxes, true_boxes, _, _, obj_acc8 = get_evaluation_bboxes_accuracy(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=CONF_THRESHOLD,
            )
            # Mean Average Precision
            print("pred box:", len(pred_boxes))
            mapval_8 = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP8: {mapval_8.item()}")

            
            # confscore 0.9
            print("=============Conf Threshold 0.9")
            CONF_THRESHOLD = 0.9
            # # Check Accuracy
            # _, _, obj_acc9 = check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            # # Non Maximum Suppression                                                 
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=CONF_THRESHOLD,
            # )                                               
            pred_boxes, true_boxes, _, _, obj_acc9 = get_evaluation_bboxes_accuracy(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=CONF_THRESHOLD,
            )
            # Mean Average Precision
            print("pred box:", len(pred_boxes))
            mapval_9 = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP9: {mapval_9.item()}")








            wandb.log({
                "Total Loss": mean_loss,
                "Box Loss" : mean_box_loss,
                "Obj Loss" : mean_obj_loss,
                "No Obj Loss" : mean_noobj_loss,
                "Class Loss" : mean_class_loss,

                # "train_class_acc" : train_class_acc, 
                # "train_noobj_acc" : train_noobj_acc, 
                # "train_obj_acc" : train_obj_acc,

                "class_acc": class_acc,
                "noobj_acc": noobj_acc,
                "obj_acc2": obj_acc,
                "obj_acc4": obj_acc4,
                "obj_acc6": obj_acc6,
                # "obj_acc7": obj_acc7,
                "obj_acc8": obj_acc8,
                "obj_acc9": obj_acc9,

                "MAP2": mapval.item(),
                "MAP4": mapval_4.item(),
                "MAP6": mapval_6.item(),
                # "MAP7": mapval_7.item(),
                "MAP8": mapval_8.item(),
                "MAP9": mapval_9.item(),
                })
            model.train()
        
        
            # print("best map:", best_map, " now map:", mapval.item(), mapval_5.item(), mapval_6.item(), mapval_7.item(), mapval_8.item(), mapval_9.item())
            print("best map:", best_map, " now map:", mapval.item(), mapval_4.item(), mapval_6.item(), mapval_8.item(), mapval_9.item())
            # print("best map:", best_map, " now map:", mapval.item())
            if config.SAVE_MODEL:
                print("model saved")
                save_checkpoint(model, optimizer, filename=f"checkpoint.pth")
            # if config.SAVE_MODEL and (best_map < mapval.item()):
            #     best_map = mapval.item()
            #     print("model saved")
            #     save_checkpoint(model, optimizer, filename=f"plain_checkpoint.pth")


if __name__ == "__main__":
    wandb.init(
        project= "object_detection_trial",
        name = "custom_full_run"
    )
    main()




    # # ================= #
    # #   scaled anchor   #
    # # ================= #
    # a = torch.tensor(config.ANCHORS)
    # b = torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    # b1 = torch.tensor(config.S).unsqueeze(1).unsqueeze(1)
    # print(a.shape)
    # print(b.shape)
    # print(a)
    # print(b)
    # scaled_anchors = a*b
    # scaled_anchors1 = a*b1
    # print(scaled_anchors.shape)
    # print(scaled_anchors1.shape)
    # print(scaled_anchors)
    # print(scaled_anchors1)

    # print(torch.tensor(config.S))
    # print(torch.tensor(config.S).unsqueeze(1))
    # print(torch.tensor(config.S).unsqueeze(1).unsqueeze(1))
    # print(torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    # print(torch.tensor(config.S).shape)
    # print(torch.tensor(config.S).unsqueeze(1).shape)
    # print(torch.tensor(config.S).unsqueeze(1).unsqueeze(1).shape)
    # print(torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).shape)