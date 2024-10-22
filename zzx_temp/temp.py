import torch
n = 1000
# 假设 data 是一个大小为 [21, n, 24] 的张量
data = torch.randn(21, n, 24)  # 示例数据

# # 计算 IC 相关系数
# # 1. 对最后一个维度进行均值和标准差计算
# mean = data.mean(dim=2, keepdim=True)  # 计算均值
# std = data.std(dim=2, keepdim=True)    # 计算标准差

# # 2. 标准化数据
# normalized_data = (data - mean) / std  # 标准化

# # 3. 计算相关系数矩阵
# # 使用矩阵乘法计算相关系数
# correlation_matrix = torch.matmul(normalized_data, normalized_data.transpose(1, 2)) / (data.size(2) - 1)

# print(correlation_matrix.shape)

# print(correlation_matrix)


def pair_wise_cos(a,b):
    # a,b: [bs, var_num, d]
    a_norm = a/a.norm(dim=2, keepdim=True)
    b_norm = b/b.norm(dim=2, keepdim=True)
    res = torch.einsum("bvd,Bvd->vbB", a_norm, b_norm)
    return res

bs = 1024
var_num = 21
d = 128
a = torch.randn(bs, var_num, d)
b = torch.randn(bs, var_num, d)

# a_norm = a/a.norm(dim=2, keepdim=True)
# b_norm = b/b.norm(dim=2, keepdim=True)
# print(a_norm.shape)
# res = torch.einsum("bvd,Bvd->vbB", a_norm, b_norm)
# print(res.shape)

mask_positives = torch.randn(bs, bs)
temp = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(bs).view(-1, 1), 0)
print(temp)


def create_tensor_with_scatter(var_num, bs):
    # 创建一个全1的张量
    tensor = torch.ones(var_num, bs, bs)
    
    # 创建对角线的索引
    indices = torch.arange(bs).unsqueeze(0)  # [1, bs]
    
    # 在每个矩阵的对角线位置上散布0
    tensor.scatter_(2, indices.unsqueeze(1), 0)

    return tensor

# 示例参数
var_num = 3  # 可以根据需要修改
bs = 4       # 可以根据需要修改

result_tensor = create_tensor_with_scatter(var_num, bs)
print(result_tensor)