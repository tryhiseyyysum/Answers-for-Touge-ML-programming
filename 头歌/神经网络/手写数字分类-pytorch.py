#encoding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import os
if os.path.exists('./step3/cnn.pkl'):
    os.remove('./step3/cnn.pkl')
    
#加载数据             
train_data = torchvision.datasets.MNIST(
    root='./step3/mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to                                                    
    download=False,
)
#取6000个样本为训练集
train_data_tiny = []

for i in range(6000):
    train_data_tiny.append(train_data[i])

train_data = train_data_tiny

#********* Begin *********#
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(  #input shape(1,28,28)
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(  #(16,14,14)
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #输出：(32,7,7)
        self.out=nn.Linear(32*7*7,10)  #10分类
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output=self.out(x)
        return output

cnn=CNN()

optimizer=torch.optim.SGD(cnn.parameters(),lr=0.01,momentum=0.9)
loss_func=nn.CrossEntropyLoss()

EPOCH=5
for epoch in range(EPOCH):
    for x,y in train_loader:
        batch_x=Variable(x)
        batch_y=Variable(y)

        output=cnn(batch_x)
        loss=loss_func(output,batch_y)

        optimizer.zero_grad()  #先将梯度清理再反向传播
        loss.backward()
        optimizer.step()


#********* End *********#
#保存模型
torch.save(cnn.state_dict(), './step3/cnn.pkl')
