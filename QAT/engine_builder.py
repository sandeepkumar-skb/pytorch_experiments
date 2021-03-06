import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image
import time
import argparse
import torch


verbose = True
if verbose:
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
else:
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(model_path, shape):
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(flags=network_flags) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_batch_size = 1
        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config = builder.create_builder_config()
        config.max_workspace_size = 1<<30
        config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
        config.flags = config.flags  | 1 << int(trt.BuilderFlag.STRICT_TYPES)

        engine = builder.build_engine(network, config)
        return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    '''
    with engine.create_execution_context() as context:  # cost time to initialize
        cuda.memcpy_htod_async(in_gpu, inputs, stream)
        context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
        cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
        stream.synchronize()
    '''

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

def transform_img(image):
    transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                    )
                                ])
    return transform(image)

if __name__ == "__main__":
    #inputs = np.random.random((1, 3, 224, 224)).astype(np.float32)
    tensors = torch.load("lenet_tensors.pth")
    inputs = tensors["input"].cpu().numpy()
    output = tensors["output"].cpu().numpy()

    #inputs = np.random.random((1, 1, 28, 28)).astype(np.float32)
    parser = argparse.ArgumentParser(description="Create a TRT Engine from ONNX export and runs infernce")
    parser.add_argument("--onnx_model", help="Provide the ONNX exported model", required=True)
    parser.add_argument("--engine", help="Provide the name of the engine to be serialized", required=False, default=None)
    args = parser.parse_args()

    inp_shape = inputs.shape
    #inp_shape = (1, 3, 224, 224)
    print("Loading {}".format(args.onnx_model))
    trt_engine = build_engine(args.onnx_model, inp_shape)
    context = trt_engine.create_execution_context()
    res = ""
    in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(trt_engine)
    for i in range(10):
        t1 = time.time()
        res = inference(trt_engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print("cost time: {:.4f}secs".format(time.time()-t1))
        #index = np.argmax(res)
    print(out_cpu.shape, type(out_cpu))
    print(output.shape, type(output))

    if args.engine is not None:
        with open(args.engine, "wb") as file:
            file.write(trt_engine.serialize())
        print("Generated the engine: {}".format(args.engine))
