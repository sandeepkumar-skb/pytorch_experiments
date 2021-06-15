# Torchvision experiments

## ResNet50
### Training
* batch_size - 128
* GPU - Tesla V100-SXM2

| Setting  | Iteration Time(ms) |
| ------------- | ------------- |
| FP32, NHWC  | 412.84  |
| FP32, NCHW  | 342.37  |
| AMP, NHWC   | 110.93 |
| AMP, NCHW   | 152.20 |

### Inference
* batch_size - 1
* GPU - Tesla V100-SXM2
* FP16 mode - model and inputs are casted to `model.half` except for TensorRT fp16

| Setting  | Iteration Time(ms) |
| ------------- | ------------- |
| FP32, NHWC  | 11.77  |
| FP32, NCHW  | 10.42  |
| FP16, NHWC   | 14.11 |
| FP16, NCHW   | 13.43 |
| TRT FP32, NCHW | 2.29 |
| TRT FP16, NCHW | 1.09 |
