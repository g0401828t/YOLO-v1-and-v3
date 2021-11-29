import config
import torch
import torch.optim as optim

import wandb
import pdb

from model import Yolov3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,get_loaders_custom,
    plot_couple_examples
)
from loss import YoloLoss_v3
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
    loss_fn = YoloLoss_v3()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv"
    )
    # train_loader, test_loader, train_eval_loader = get_loaders_custom(
    #     train_csv_path=config.DATASET + "/100examples.csv", test_csv_path=config.DATASET + "/100examples.csv"
    # )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # [3] => [3,1] => [3,1,1] => [3,3,2]
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        print("=====epochs:",epoch,"======")
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        # train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)


        mean_loss, mean_box_loss, mean_obj_loss, mean_noobj_loss, mean_class_loss \
            = train_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)  # for testing train without augmentations

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # validation
        if epoch > 0 and epoch % 10 == 0:
            class_acc, noobj_acc, obj_acc = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
          # Non Maximum Suppression
            pred_boxes, true_boxes = get_evaluation_bboxes(
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
            print(f"MAP: {mapval.item()}")
            wandb.log({
                "Total Loss": mean_loss,
                "Box Loss" : mean_box_loss,
                "Obj Loss" : mean_obj_loss,
                "No Obj Loss" : mean_noobj_loss,
                "Class Loss" : mean_class_loss,
                "class_acc": class_acc,
                "noobj_acc": noobj_acc,
                "obj_acc": obj_acc,
                "MAP": mapval.item()})
            model.train()
        
        
            if config.SAVE_MODEL and mapval.item() > 0.99:
                save_checkpoint(model, optimizer, filename=f"checkpoint.pth")


if __name__ == "__main__":
    wandb.init()
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