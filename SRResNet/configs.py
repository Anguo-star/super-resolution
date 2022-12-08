import torch


# 全局参数
RANDOM_SEED = 6  # 设置随机种子

# 数据集参数
LFW_IMAGE_PATH = r'data/lfw'  # 数据集相对根路径
IMAGE_SIZE = 256  # 输入统一调整尺寸
NUM_WORKERS = 4

# 模型参数
MODEL_NAME = 'SRResNet'  # 定义选择哪个模型
NUM_RESIDUAL_LAYERS = 16  # 残差块包含的残差层数量
UPSCALE_FACTOR = 4  # 放大倍数（必须为2的整数倍）
INPUT_CHANNELS = 3  # 网络输入通道数
FEATURE_MAP_CHANNELS = 64  # 中间卷积层输出feature_map通道数
OPEN_CHECKPOINT = True
CHECKPOINT_PATH = r'results/saved_model/SRResNet.pth'
MODEL_PATH = r'results/saved_model/SRResNet.pth'
HISTORY_PATH = r'results/history/SRResNet.npy'

# 训练参数
BATCH_SIZE = 128  # 数据集批量尺寸
NUM_EPOCHS = 200  # 训练迭代次数
LOSS_FN = 'MSE'  # 损失函数
OPTIMIZER = 'adam'  # 优化方法
LEARNING_RATE = 1e-3  # 初始学习率
LR_SCHEDULE = 'StepLR'  # 学习率策略
LR_GAMA = 0.98  # gama衰减参数
STEP_SIZE = 5  # StepLR策略下的变化步长
MILESTONES = [50, 100, 110, 120, 150]  # MultiStepLR策略下学习率变化点集合
COS_T_MAX = 50  # 余弦退火策略周期
COS_ETA_MIN = 1e-7  # 余弦退火策略最小值
LR_LAMBDA = lambda epoch: LEARNING_RATE * (NUM_EPOCHS - epoch) / NUM_EPOCHS  # 自定义学习策略

# 设备参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先选择gpu，若无gpu默认使用cpu
NGPU = torch.cuda.device_count()  # gpu数量
