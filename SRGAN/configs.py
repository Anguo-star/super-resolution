import torch


# 全局参数
RANDOM_SEED = 6  # 设置随机种子

# 数据集参数
LFW_IMAGE_PATH = r'data/lfw_flatten'  # 数据集相对根路径
IMAGE_SIZE = 256  # 输入统一调整尺寸
NUM_WORKERS = 2

# 模型参数
GENERATOR = 'SRResNet'  # 定义选择生成模型
DISCRIMINATOR = 'SRGAN'  # 定义GAN判别模型
NUM_RESIDUAL_LAYERS = 4  # 残差块包含的残差层数量
UPSCALE_FACTOR = 4  # 放大倍数（必须为2的整数倍）
INPUT_CHANNELS = 3  # 网络输入通道数
FEATURE_MAP_CHANNELS = 64  # 中间卷积层输出feature_map通道数
OPEN_CHECKPOINT_READ = True  # 开启读checkpoint
OPEN_CHECKPOINT_WRITE = True  # 开启写checkpoint
CHECKPOINT_PATH = r'../../../../../working/results/saved_model/SRResNet_256_128x2_4res.pth'  # 若用kaggle训练，路径前需加 ../../../../../working/
MODEL_PATH = r'../../../../../working/results/saved_model/SRResNet_256_128x2_4res.pth'  # 若用kaggle训练，路径前需加 ../../../../../working/
HISTORY_PATH = r'../../../../../working/results/history/SRResNet_256_128x2_4res.npy'  # 若用kaggle训练，路径前需加 ../../../../../working/

# 训练参数
BATCH_SIZE = 64  # 数据集批量尺寸
NUM_EPOCHS = 300  # 训练迭代次数
LOSS_FN = 'MSE'  # 损失函数
OPTIMIZER = 'adam'  # 优化方法
LEARNING_RATE = 1e-3  # 初始学习率
LR_SCHEDULE = 'StepLR'  # 学习率策略, 可选策略有['StepLR', 'MultistepLR', 'ExponentialLR', 'CosineAnnealingLR', 'LambdaLR']
LR_GAMA = 0.9  # gama衰减参数
STEP_SIZE = 10  # StepLR策略下的变化步长
MILESTONES = [50, 100, 110, 120, 150]  # MultiStepLR策略下学习率变化点集合
COS_T_MAX = 50  # 余弦退火策略周期
COS_ETA_MIN = 1e-7  # 余弦退火策略最小值
LR_LAMBDA = lambda epoch: LEARNING_RATE * (NUM_EPOCHS - epoch) / NUM_EPOCHS  # 自定义学习策略

# 设备参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先选择gpu，若无gpu默认使用cpu
NGPU = torch.cuda.device_count()  # gpu数量
