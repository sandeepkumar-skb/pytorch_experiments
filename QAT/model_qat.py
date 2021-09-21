import torch
import torch.nn as nn
from checker import checker
from minmaxobserver import MinMaxObserver

class QConv2d(nn.Conv2d):
    def __init__(self, *args, activation=nn.ReLU(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.fq_node = torch.quantization.FakeQuantize(observer=MinMaxObserver)

    def forward(self, x):
        qweight = self.fq_node(self.weight)
        return self.activation(self._conv_forward(x, qweight, self.bias))

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = QConv2d(1, 1, 1, activation=torch.nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval().cuda()
input_fp32 = torch.randn(4, 1, 4, 4).cuda()
res = model_fp32(input_fp32)
print("Eval Done")
checker(model_fp32, input_fp32)
