a = [1,2,3]
b = a

if True:
    a = [4, 5, 6]

print("a:", a)
print("b:", b)  
import torch
batch = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shape = (batch, 3, 64, 64)  # Example shape for an image tensor
a = torch.randn(shape, device=device)
print(a.shape)
b = a.repeat(10, 1, 1, 1)
print(b.shape)
print(b)


# 示例
alphas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # 假设时间长度为5
t_batch = torch.tensor([2, 4])  # 表示两个样本的时间步是2和4
result = alphas[t_batch]  # => tensor([0.3, 0.5])
print("result:", result)

list = [1, 2, 3, 4, 5]

def fun1(list):
    
    list = [x + 1 for x in list]
    
    return list

def fun2(list):
    list = [x + 1 for x in list]
    return list

middle = fun1(list)
result = fun2(middle)
print("middle:", middle)
print("result:", result)

