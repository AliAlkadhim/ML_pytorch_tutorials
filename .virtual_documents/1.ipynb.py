import torch
import pandas as pd
# to convert from df format to tensor format, do torch.tensor(df[something].values)



A = torch.arange(20, dtype=torch.float32).reshape(5, 4); A


B = A.clone() ; B


A * B


x = torch.arange(4, dtype=torch.float32)
x, x.sum()


A.shape, A.sum()


C = A.sum(); C


A.numel()


sum_A = A.sum(axis=1, keepdims=True)
sum_A


B = torch.ones(4, 3)
torch.mm(A, B)
\



