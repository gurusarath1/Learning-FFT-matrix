import torch
import torch.nn as nn

class fft_model(nn.Module):

    def __init__(self, fft_size):
        super(fft_model, self).__init__()

        self.fft_size = fft_size

        # FFT is a linear transformation not affine
        self.model = nn.Sequential(
            nn.Linear(self.fft_size, self.fft_size * 2, bias=False)
        )

    def forward(self, x):

        return self.model(x)