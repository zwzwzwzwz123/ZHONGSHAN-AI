import torch
print(torch.__version__)  # 确保输出 ≥1.9.0

# 验证函数是否存在
if hasattr(torch, 'spsolve'):
    print("spsolve is available")
else:
    print("需升级PyTorch或检查安装")