import torch
import torch.nn as nn
import torchvision.models as models
import time as time
torch.backends.cudnn.benchmark=True


# ARGS
use_amp=True
use_nhwc=True
dtype=torch.float32
inference_mode=False
batch_size=128


def run_inference(model, inp):
    print("Running Inference")
    model.eval().cuda()
    if dtype == torch.half:
        print("Converting model to half")
        print("Input type: {}".format(inp.dtype))
        model.half()
    with torch.no_grad():
        for i in range(100):
            start = time.time()
            out = model(inp)
            torch.cuda.synchronize()
            stop = time.time()
            print("Iter Time for Iter {} is: {:.2f}ms".format(i, (stop-start)*1000))

def run_training(model, inp):
    print("Running Training")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)
    model.cuda().train()
    target = torch.ones([batch_size], device='cuda', dtype=torch.long)

    for i in range(100):
        with torch.cuda.amp.autocast(enabled=use_amp):
            start = time.time()
            out = model(inp)
            loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        print("Training iteration time for Iter {} is {:.2f}ms".format(i, (end-start)*1000))





if __name__ == "__main__":
  
    model = models.resnet50(pretrained=False)

    inp = torch.randn((batch_size, 3, 224,224), device='cuda', dtype=dtype)
    if use_nhwc:
        model = model.to(memory_format=torch.channels_last)
        inp = inp.to(memory_format=torch.channels_last)

    if inference_mode:
        run_inference(model, inp)
    else:
        run_training(model, inp)

    print("Done")
