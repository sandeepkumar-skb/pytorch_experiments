# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = torch.nn.intrinsic.qat.ConvReLU2d(1, 1, 1, qconfig=torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8), weight=torch.quantization.default_observer.with_args(dtype=torch.qint8)))

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.conv(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval().cuda()
input_fp32 = torch.randn(4, 1, 4, 4).cuda()
res = model_fp32(input_fp32)
torch.onnx.export(model_fp32, input_fp32, "model_qat.onnx", opset_version=13)
print(model_fp32)

model_jit = torch.jit.script(model_fp32)
torch.jit.save(model_jit, "model_jit.pth")
