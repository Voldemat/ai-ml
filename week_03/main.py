import torch

_ = torch.manual_seed(42)

scalar = torch.tensor(1)
print(f"{scalar=}")
vector = torch.tensor([1, 2, 3])
print(f"{vector=}")
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(f"{matrix=}")
transposed = matrix.T
print(f"{transposed=}")

linear = torch.nn.Linear(in_features=3, out_features=3)
output = linear(matrix)
print(f"linear output: {output}")

x = torch.arange(0, 100, 10)
print(f"{x=}")
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Mean: {x.type(torch.float32).mean()}")
print(f"Sum: {x.sum()}")

print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")

print(f"x.reshape: {x.reshape(1, 10)}")
print(f"x.view: {x.view(1, 10)}")

print(f"torch.stack: {torch.stack([x, torch.arange(100, 200, 10)])}")
print(f"x.reshape.squeeze: {x.reshape(1, 10).squeeze()}")
assert x.reshape(1, 10).squeeze().equal(x)

print(f"x.unsqueeze(0): {x.unsqueeze(0)}")
print(f"x.unsqueeze(1): {x.unsqueeze(1)}")

y = torch.rand(size=(224, 225, 3))

y_permutated = y.permute(2, 0, 1)

print(f"y shape: {y.shape}")
print(f"y_permutated shape: {y_permutated.shape}")

x_gpu = x.to("mps")
print(f"x_gpu: {x_gpu}")
x_cpu = x_gpu.cpu()
print(f"x_cpu: {x_cpu}")
