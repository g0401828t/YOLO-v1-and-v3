"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from torch.nn.modules.activation import Softshrink
from utils import intersection_over_union


class YoloLoss_v1(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # ious_b1, ious_b2: (batch_size, 7, 7, 4)
        # ious: (2, batch_size, 7, 7, 4)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        #iou_maxes: (batch_size, 7, 7, 1)
        # bestbox: (batch_size, 7, 7, 1)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        # exists_box: (batch_size, 7, 7, 1)
        # exists_box = target[..., 20].unsqueeze(3)  # (batch_size, 7, 7) -> (batch_size, 7, 7, 1)
        exists_box = target[..., 20:21]  # same as unsqueezing


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) \
                                    * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        
        # ================== #
        #     TOTAL LOSS     #
        # ================== #

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss









# ============================ #
#           Yolo v3            #
# ============================ #

# Loss for single scale
class YoloLoss_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.smoothl1 = nn.SmoothL1Loss(beta=0.05)

        # # constants => mse for box_loss
        # self.lambda_class = 1
        # self.lambda_noobj = 10
        # self.lambda_obj = 1
        # # self.lambda_box = 10
        # # self.lambda_box = 1      # box 가 믾아쟈사 box_loss값이 커져서 낮춰주자. 0.01은 너무 작아서 10 -> 0.01 -> 1
        # self.lambda_box = 5      # box 가 믾아쟈사 box_loss값이 커져서 낮춰주자. 0.01은 너무 작아서 10 -> 0.01 -> 1 -> 5

        # # constants => ciou for box_loss
        # self.lambda_class = 0.5
        # self.lambda_noobj = 10
        # self.lambda_obj = 1
        # self.lambda_box = 0.5

        # # constants => mse for box_loss
        # self.lambda_class = 0.5
        # self.lambda_obj = 1
        # self.lambda_box = 0.05

        # # constants => mse for box_loss
        # self.lambda_class = 1
        # self.lambda_noobj = 10
        # self.lambda_obj = 500      # objloss 가 증가 한다 pred를 다 0으로 해야 전체 loss가 줄어든다고 생각해서 이 loss 증가한다고 판단하여 1 에서 100으로 바꿈 -> 500
        # self.lambda_box = 5      # box 가 믾아쟈사 box_loss값이 커져서 낮춰주자. 0.01은 너무 작아서 10 -> 0.01 -> 1 -> 5
        
        # constants => mse for box_loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1      
        self.lambda_box = 10      

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1           #  [B, 3, S, S]
        noobj = target[..., 0] == 0
        # noobj = (target[..., 0] < 1) * (target[...,0] >= 0)
        soft_mask = (target[...,0] > 0 ) * (target[...,0] < 1)
        # cls_mask = target[..., 0] >= 0.9




        anchors = anchors.reshape(1, 3, 1, 1, 2)  # (3, 2) -> (1, 3, 1, 1, 2)

        # # ========================= #
        # #       No object loss      #   
        # # ========================= #
        # no_object_loss = self.bce(
        #     (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
        # )
        # no_object_loss = self.lambda_noobj * no_object_loss

        # # ========================= #
        # #        Object loss        # 
        # # ========================= #
        # anchors = anchors.reshape(1, 3, 1, 1, 2)  # (3, 2) -> (1, 3, 1, 1, 2)  
        # box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)        # for calculating p_w * exp(t_w)
        # ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        # # object_loss = self.bce((predictions[..., 0:1][obj]), (target[..., 0:1][obj]))                         # without iou computations

        # # object_loss = self.bce((predictions[..., 0:1][soft_mask]), (target[..., 0:1][soft_mask]))           # without iou computations (soft mask)

        # ========================= #
        #       Ours Object Loss    #
        # ========================= #  
        # soft_ce_noobj = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))                            # obj == 0
        # no_object_loss = soft_ce_noobj
        # # trial 1
        # soft_ce_obj = -target[..., 0:1][soft_mask] * torch.log(self.sigmoid(predictions[..., 0:1][soft_mask]) + 1e-6)  # obj > 0
        # soft_ce_obj = soft_ce_obj.mean()
        # # trail 2
        # soft_ce_obj = self.mse(target[..., 0:1][soft_mask], self.sigmoid(predictions[..., 0:1][soft_mask]))  # obj > 0
        # # trail 3
        # soft_ce_obj = -torch.log(self.sigmoid(predictions[..., 0:1][soft_mask]) - target[...,0:1][soft_mask] + 1 + 1e-6)  # obj > 0
        # soft_ce_obj = soft_ce_obj.mean()
        # # trial 4
        # soft_ce_obj = self.smoothl1(target[..., 0:1][soft_mask], self.sigmoid(predictions[..., 0:1][soft_mask]))  # obj > 0

        # # trial 5 : modified bce
        # # obj loss
        # box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)        
        # ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        # # noobj loss + soft_loss
        # noobj_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))           
        # soft_loss = -1 *  (target[..., 0:1][soft_mask] * torch.log(self.sigmoid(predictions[..., 0:1][soft_mask]) + 1e-6) + torch.log(1 - self.sigmoid(predictions[..., 0:1][soft_mask]) + 1e-6))
        # soft_loss = soft_loss.mean()
        # no_object_loss = self.lambda_noobj * noobj_loss + soft_loss

        # # trial 6 : softing on no obj loss 제발 되어라...
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)        
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        
        noobj_loss = -(1-target[...,0:1][noobj]) * torch.log(1 - self.sigmoid(predictions[..., 0:1][noobj]) + 1e-6)
        soft_noobj_loss = -(1-target[...,0:1][soft_mask]) * torch.log(1 - self.sigmoid(predictions[..., 0:1][soft_mask]) + 1e-6)
        no_object_loss = self.lambda_noobj * (noobj_loss.mean() + soft_noobj_loss.mean())



        # ========================= #
        #    Box Coordinate loss    # 
        # ========================= # 
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(                             # for calculating: w = p_w * exp(t_w)
            (1e-6 + target[..., 3:5] / anchors)
        )
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])                                  # obj == 1
        # ciou_loss = intersection_over_union(predictions[..., 1:5][obj], target[..., 1:5][obj], CIOU=True)
        # box_loss = (1 - ciou_loss).mean()


        # ========================= #
        #        Class loss         #   
        # ========================= #
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        # class_loss = self.entropy(
        #     (predictions[..., 5:][cls_mask]), (target[..., 5][cls_mask].long()),                                # obj == 0.9
        # )

        # return(
        #     self.lambda_box * box_loss
        #     + self.lambda_obj * object_loss
        #     + self.lambda_noobj * no_object_loss
        #     + self.lambda_class * class_loss
        # )
        # return self.lambda_box * box_loss, self.lambda_obj * object_loss, self.lambda_noobj * no_object_loss, self.lambda_class * class_loss
        return self.lambda_box * box_loss, self.lambda_obj * object_loss, no_object_loss, self.lambda_class * class_loss


if __name__ == "__main__":
    # ==================== #
    #  masking obj, noobj  #
    # ==================== #
    a = torch.zeros(3, 5, 5, 7)
    a[0, 2, 3, 0] = 1
    a[1, 2, 3, 0] = 1
    a[2, 2, 3, 0] = 1
    obj = a[..., 0] == 1
    noobj = a[..., 0] == 0

    pred = torch.rand(3, 5, 5, 7)

    print(pred[..., 0])
    print(obj)
    print(pred[..., 0:1][obj])
    print(pred[..., 0:1][noobj])