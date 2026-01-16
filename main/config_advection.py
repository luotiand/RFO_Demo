import torch
from rectified.rectified_flow import RectFlow
# 参数设置
para_path = "/data5/store1/dlt/rectified_flow/modelpara/"
save_path = "/data5/store1/dlt/rectified_flow/result/advecion/"
eq_T = 1.0 # 方程采样总长
N = 10000  # a数量
niter = 500
target_len=512
lr = 1e-3
batch_size = 128  # 减少批处理大小
T = 1.0
rf_dt = 1.0
h_dim = 4096
train = 0
rf = RectFlow()
# 模型相关
scorenet_model_class = "FNO1d"  

# GPU 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 文件名自动生成
model_name = scorenet_model_class.lower() + ".pth"
