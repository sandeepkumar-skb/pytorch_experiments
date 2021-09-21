import torch
import torch.nn as nn
from minmaxobserver import MinMaxObserver

from checker import checker

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

fq_weight = torch.quantization.FakeQuantize.with_args(\
    observer=MinMaxObserver,
    quant_min=0, quant_max=255, dtype=torch.quint8)

fq_activation = torch.quantization.FakeQuantize.with_args(\
    observer=MinMaxObserver,
    quant_min=0, quant_max=255, dtype=torch.quint8)


model = LeNet().cuda()
model.conv.qconfig = torch.quantization.QConfig(activation=fq_activation, weight=fq_weight)
torch.quantization.prepare_qat(model, inplace=True)
#model.l1.apply(torch.quantization.disable_fake_quant)
model.conv.apply(torch.quantization.disable_fake_quant)
input_fp32 = torch.rand(1, 3, 32, 32, device='cuda')
print(model)
torch.onnx.export(model, input_fp32, "prepare_qat.onnx", opset_version=10)

checker(model, input_fp32)
