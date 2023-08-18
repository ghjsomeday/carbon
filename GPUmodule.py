import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 添加tensorboard
writer = SummaryWriter("logs_trainGPU")

# 模型构建
device = torch.device("mps")
tudui = torch.load("vgg16_method1.pth")
tudui = tudui.to(device)

# 损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr = learning_rate)

# 训练次数记录
train_step = 0
test_step = 0
epoch = 10

# 开始训练
for i in range(epoch):
    print("————第{}轮训练开始————".format(i+1))

    tudui.train()
    for data in train_loader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step = train_step + 1
        if train_step % 100 == 0:
            print("训练次数为{}时，损失为{}".format(train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),train_step)

    # 每轮的测试
    tudui.eval()
    total_test_loss = 0
    total_accuarcy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuarcy = total_accuarcy + accuracy
    print ("整体测试集上的损失为{},正确率为{}".format(total_test_loss,total_accuarcy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,test_step)
    writer.add_scalar("accuarcy",total_accuarcy,test_step)
    test_step = test_step + 1

writer.close()