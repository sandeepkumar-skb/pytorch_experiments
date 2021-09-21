import torch

class MinMaxObserver(torch.quantization.MinMaxObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_orig):
        if not self.training: return x_orig
        return super().forward(x_orig)
