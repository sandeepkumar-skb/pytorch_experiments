import torch
import torch
from pytorch_quantization import nn as quant_nn
quant_nn.TensorQuantizer.use_fb_fake_quant = True

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = quant_nn.QuantConv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval().cuda()
input_fp32 = torch.randn(4, 1, 4, 4).cuda()
res = model_fp32(input_fp32)
print("Done")
print(model_fp32)
torch.onnx.export(model_fp32, input_fp32, "model_nvqat.onnx", opset_version=10)
