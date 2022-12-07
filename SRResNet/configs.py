import torch


# 全局参数
RANDOM_SEED = 6  # 设置随机种子

# 数据集参数
LFW_IMAGE_PATH = r'data/lfw'  # 数据集相对根路径
IMAGE_SIZE = 256  # 输入统一调整尺寸
NUM_WORKERS = 2

# 模型参数
MODEL_NAME = 'SRResNet'  # 定义选择哪个模型
NUM_RESIDUAL_LAYERS = 16  # 残差块包含的残差层数量
UPSCALE_FACTOR = 4  # 放大倍数（必须为2的整数倍）
INPUT_CHANNELS = 3  # 网络输入通道数
FEATURE_MAP_CHANNELS = 64  # 中间卷积层输出feature_map通道数
OPEN_CHECKPOINT = True
CHECKPOINT_PATH = r'results/saved_model/SRResNet_256_64x4_16res.pth'
MODEL_PATH = r'results/saved_model/SRResNet_256_64x4_16res.pth'
HISTORY_PATH = r'results/history/SRResNet_256_64x4_16res.npy'

# 学习参数
BATCH_SIZE = 32  # 数据集批量尺寸
NUM_EPOCHS = 200  # 训练迭代次数
LEARNING_RATE = 1e-3  # 初始学习率
LOSS_FN = 'MSE'  # 损失函数
OPTIMIZER = 'adam'  # 优化方法

# 设备参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先选择gpu，若无gpu默认使用cpu
NGPU = torch.cuda.device_count()  # gpu数量

