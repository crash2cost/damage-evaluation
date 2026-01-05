import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels, kernel_size=3)
        self.cv2 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.add = shortcut and in_channels == out_channels
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = ConvBlock(in_channels, 2 * hidden_channels, kernel_size=1)
        self.cv2 = ConvBlock((2 + n) * hidden_channels, out_channels, kernel_size=1)
        self.m = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut) 
            for _ in range(n)
        )
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBlock(hidden_channels * 4, out_channels, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class YOLOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBlock(3, 16, kernel_size=3, stride=2)
        self.stage1_conv = ConvBlock(16, 32, kernel_size=3, stride=2)
        self.stage1_c2f = C2f(32, 32, n=1)
        self.stage2_conv = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.stage2_c2f = C2f(64, 64, n=2)
        self.stage3_conv = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.stage3_c2f = C2f(128, 128, n=2)
        self.stage4_conv = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.stage4_c2f = C2f(256, 256, n=1)
        self.sppf = SPPF(256, 256)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1_conv(x)
        x = self.stage1_c2f(x)
        x = self.stage2_conv(x)
        p3 = self.stage2_c2f(x)
        x = self.stage3_conv(p3)
        p4 = self.stage3_c2f(x)
        x = self.stage4_conv(p4)
        x = self.stage4_c2f(x)
        p5 = self.sppf(x)
        return p3, p4, p5
class YOLONeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(384, 128, n=1)
        self.c2f_p3 = C2f(192, 64, n=1)
        self.conv_n3 = ConvBlock(64, 64, kernel_size=3, stride=2)
        self.c2f_n4 = C2f(192, 128, n=1)
        self.conv_n4 = ConvBlock(128, 128, kernel_size=3, stride=2)
        self.c2f_n5 = C2f(384, 256, n=1)
    def forward(self, features):
        p3, p4, p5 = features
        x = self.upsample(p5)
        x = torch.cat([x, p4], dim=1)
        x = self.c2f_p4(x)
        p4_out = x
        x = self.upsample(x)
        x = torch.cat([x, p3], dim=1)
        p3_out = self.c2f_p3(x)
        x = self.conv_n3(p3_out)
        x = torch.cat([x, p4_out], dim=1)
        n4_out = self.c2f_n4(x)
        x = self.conv_n4(n4_out)
        x = torch.cat([x, p5], dim=1)
        n5_out = self.c2f_n5(x)
        return p3_out, n4_out, n5_out
class DetectionHead(nn.Module):
    def __init__(self, num_classes=1, channels=(64, 128, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.nl = len(channels)
        self.reg_max = 16
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, 64, kernel_size=3),
                ConvBlock(64, 64, kernel_size=3),
                nn.Conv2d(64, 4 * self.reg_max, kernel_size=1)
            ) for c in channels
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, 64, kernel_size=3),
                ConvBlock(64, 64, kernel_size=3),
                nn.Conv2d(64, num_classes, kernel_size=1)
            ) for c in channels
        )
    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            box = self.cv2[i](feat)
            cls = self.cv3[i](feat)
            outputs.append(torch.cat([box, cls], dim=1))
        return outputs
class CustomYOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = YOLOBackbone()
        self.neck = YOLONeck()
        self.head = DetectionHead(num_classes=num_classes)
    def forward(self, x):
        features = self.backbone(x)
        fused = self.neck(features)
        detections = self.head(fused)
        return detections
    def load_pretrained_weights(self, pretrained_model_path):
        from ultralytics import YOLO
        official_model = YOLO(pretrained_model_path)
        official_state = official_model.model.state_dict()
        our_state = self.state_dict()
        matched_weights = {}
        for our_key in our_state.keys():
            for official_key in official_state.keys():
                if our_state[our_key].shape == official_state[official_key].shape:
                    matched_weights[our_key] = official_state[official_key]
                    break
        self.load_state_dict(matched_weights, strict=False)
        print(f"Loaded {len(matched_weights)}/{len(our_state)} weights from pretrained model")
if __name__ == "__main__":
    model = CustomYOLO(num_classes=1)
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Detection layer {i+1} output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")