import torch
import torch.nn as nn


# =============================== #
#            Yolo v1              #
# =============================== #
""" 
Information about architecture config:
Tuple : conv (kernel_size, filters, stride, padding) 
"M" : maxpooling with stride 2x2 and kernel 2x2
List : tuples (conv) and last int (number of repeats)
"""

yolov1_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self._use_bn_act = bn_act

    def forward(self, x):
        if self._use_bn_act:
            return self.leakyrelu(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = yolov1_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels = in_channels,
                            out_channels = conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels = conv1[1],
                            out_channels = conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), 
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )



# =============================== #
#            Yolo v3              #
# =============================== #
""" 
Information about architecture config:
Tuple : conv (out_channels, kernel_size, stride, padding) 
List : "B" for block and int for number of repeats
Str : "S" for output branch, "U" for upsampling
"""
yolov3_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],           # skip connection 2
    (512, 3, 2),
    ["B", 8],           # skip connection 1
    (1024, 3, 2),
    ["B", 4],           # To this point is Darknet-53 (pretrained in imagenet)
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",                # skip connection 1
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",               # skip connection 1
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                CNNBlock(channels, channels//2, kernel_size=1),
                CNNBlock(channels//2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)   # 3 anchors boxes per cell, [0, 1, 2, ..., C, p0, x, y, w, h]
        )
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])  # (N, S, S, (num_classes + 5)*3) -> (N, 3, num_classes+5, S, S)
            .permute(0, 1, 3, 4, 2)  # (N, 3, num_classes+5, S, S) -> (N, 3, S, S, num_classes+5)
        )


class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:               
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x) 

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in yolov3_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size = kernel_size,
                        stride =stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats
                    )
                )
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels//2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3  # for concatenation
        
        return layers
            





def test():
    # ================= #
    #      yolo v1      #
    # ================= #
    BATCH_SIZE = 2
    IMAGE_SIZEv1 = 448
    x1 = torch.randn(BATCH_SIZE, 3, IMAGE_SIZEv1, IMAGE_SIZEv1)

    split_size=7
    num_boxes=2
    num_classes=20
    model_yolov1 = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)

    out1 = model_yolov1(x1)
    
    print("output size:", out1.shape)
    assert out1.shape == torch.Size([BATCH_SIZE, split_size * split_size * (num_classes + num_boxes * 5)])
    print("YOlO v1 success!")



    # ================= #
    #      yolo v3      #
    # ================= #
    BATCH_SIZE = 2
    NUM_ANCHORS = 3
    IMAGE_SIZEv3 = 416
    x2 = torch.randn(BATCH_SIZE, 3, IMAGE_SIZEv3, IMAGE_SIZEv3)
    
    model_yolov3 = Yolov3(num_classes=num_classes)
    
    out2 = model_yolov3(x2)

    print("outputs size: ")
    print("output0:", out2[0].shape)
    print("output1:", out2[1].shape)
    print("output2:", out2[2].shape)
    assert out2[0].shape == torch.Size([BATCH_SIZE, NUM_ANCHORS, IMAGE_SIZEv3//32, IMAGE_SIZEv3//32, (num_classes + 5)])
    assert out2[1].shape == torch.Size([BATCH_SIZE, NUM_ANCHORS, IMAGE_SIZEv3//16, IMAGE_SIZEv3//16, (num_classes + 5)])
    assert out2[2].shape == torch.Size([BATCH_SIZE, NUM_ANCHORS, IMAGE_SIZEv3//8, IMAGE_SIZEv3//8, (num_classes + 5)])
    print("YOlO v3 success!")


    import pytorch_model_summary

    net = Yolov3(num_classes=num_classes)
    print(pytorch_model_summary.summary(net, torch.zeros(1, 3, 416, 416), show_input=False))

    # from torchviz import make_dot
    # img = torch.zeros(1, 3, 416, 416)
    # make_dot(net(img), params=dict(list(net.named_parameters())))

    from torchinfo import summary
    summary(net, input_size=(1, 3, 416, 416))

if __name__ == "__main__":
    test()