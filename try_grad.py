import torch

torch.manual_seed(123)
x = torch.Tensor([3])

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y = torch.mul(w, x)
z = torch.add(y, b)

y.backward()
print("yes")