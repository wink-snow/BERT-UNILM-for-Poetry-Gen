import torch
print(torch.__version__)
# 尝试导入
try:
    from torch.amp import GradScaler
    print("torch.amp.GradScaler found")
except ImportError:
    print("torch.amp.GradScaler NOT found")

try:
    from torch.cuda.amp import GradScaler
    print("torch.cuda.amp.GradScaler found")
except ImportError:
    print("torch.cuda.amp.GradScaler NOT found")