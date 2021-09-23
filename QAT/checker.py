import torch

def checker(model, inp):
    #print(model)
    onnx_op10_convert = "PASS"
    onnx_op13_convert = "PASS"
    jit_script_convert = "PASS"
    jit_trace_convert = "PASS"

    try:
        torch.onnx.export(model, inp, "model_op10.onnx", opset_version=10)
    except:
        onnx_op10_convert = "FAIL"

    try:
        torch.onnx.export(model, inp, "model_op13.onnx", opset_version=13)
    except:
        onnx_op13_convert = "FAIL"


    try:
        model_scripted = torch.jit.script(model)
    except:
        jit_script_convert = "FAIL"

    try:
        model_jit = torch.jit.trace(model, inp)
    except:
        jit_trace_convert = "FAIL"


    print(f"ONNX op10 export status: {onnx_op10_convert}")
    print(f"ONNX op13 export status: {onnx_op13_convert}")
    print(f"JIT script convert status: {jit_script_convert}")
    print(f"JIT trace convert status: {jit_trace_convert}")
