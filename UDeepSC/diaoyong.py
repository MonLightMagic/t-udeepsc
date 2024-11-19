# 导入模型
from model import UDeepSC_model, UDeepSC_new_model

# # 初始化模型
# model1 = UDeepSC_model(pretrained=False)  # 不加载预训练权重
# # model2 = UDeepSC_new_model(pretrained=False)  # 不加载预训练权重

# # 若需要加载预训练权重，可以传入 init_ckpt 参数
# # model1 = UDeepSC_model(pretrained=True, init_ckpt='path/to/your/checkpoint.pth')

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # model = UDeepSC_model(pretrained=False, **kwargs)


# num_params = count_parameters(model1)
# print(f"The model has {num_params} trainable parameters.")


def count_parameters(model):
    total_params = 0
    params_per_module = {}

    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            params_per_module[name] = num_params
            total_params += num_params

    return total_params, params_per_module

# 创建模型实例
model1 = UDeepSC_model(pretrained=False)

# 计算参数数量
total_params, params_per_module = count_parameters(model1)

print(f"总参数数量: {total_params}")
print("各模块参数数量:")
for module_name, num in params_per_module.items():
    print(f"{module_name}: {num}")
