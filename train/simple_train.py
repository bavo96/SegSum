import torch
from torchinfo import summary

from model.summarizer import Sum

model = Sum(input_size=1024, output_size=1024, block_size=2, attn_mechanism=False)
summary(model, [(100, 1024), (27, 2)], dtypes=[torch.float, torch.float], device="cpu")
