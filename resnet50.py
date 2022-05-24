#%%
from albumentations.augmentations.transforms import RandomBrightnessContrast
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules.loss import BCELoss

class block(nn.Module):
    def __init__(self,in_channels,intermediate_channels,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels*self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self,x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class Resnet(nn.Module):
    def __init__(self,block,layers,image_channels,num_classes):
        super(Resnet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block,layers[0],intermediate_channels=64,stride=1)
        self.layer2 = self._make_layer(block,layers[1],intermediate_channels=128,stride=2)
        self.layer3 = self._make_layer(block,layers[2],intermediate_channels=256,stride=2)
        self.layer4 = self._make_layer(block,layers[3],intermediate_channels=512,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x
    def _make_layer(self,block,num_residual_blocks,intermediate_channels,stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                self.in_channels,
                intermediate_channels*4,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(intermediate_channels*4),
            )
        layers.append(block(self.in_channels,intermediate_channels,identity_downsample,stride))

        self.in_channels = intermediate_channels*4

        for i in range(num_residual_blocks -1):
            layers.append(block(self.in_channels,intermediate_channels))
        return nn.Sequential(*layers)

def Resnet50(img_channels=3,num_classes=2):
    return Resnet(block,[3,4,6,3],img_channels,num_classes)

model = Resnet50()
y = model(torch.randn(4,3,1360,1024))
print(y.size())




# %%
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import pathlib
import random
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Mydataset(Dataset):
    def __init__(self,dataset_path):
        d_path = pathlib.Path(dataset_path)
        self.img_path_list = list(d_path.glob("**/*.jpg"))
        random.shuffle(self.img_path_list)
        self.transforms = A.Compose([
			A.Resize(256,340),#Resize(height,width)
			A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
			A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255),
            RandomBrightnessContrast(),
			ToTensorV2()
		])
        self.len = len(self.img_path_list)
        self.label_list = ["ca","pan"]

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        img = Image.open(self.img_path_list[idx])
        img = np.array(img)
        stem = str(self.img_path_list[idx].parts[-2])
        label = float(self.label_list.index(stem))
        augmentated = self.transforms(image=img)["image"]

        return augmentated , torch.tensor(label)

#%%
Batchsize = 16
dataset = Mydataset(r"D:\STUDY\ca_pan_seg\dataset")
print(len(dataset))
train_dataset = Subset(dataset,list(range(0,int(len(dataset)*0.8))))
val_dataset = Subset(dataset,list(range(int(len(dataset)*0.8),len(dataset))))

train_dataloader = DataLoader(train_dataset,batch_size=Batchsize,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=Batchsize,shuffle=False)

#%%
#train_d = iter(train_dataloader)
#print(next(train_d))
# %%
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt

device = torch.device("cuda")
BCE_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-5,betas=(0.5,0.999))

epoch = 10
model.to(device)
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
#%%
for i in range(epoch):
    print("epoch:{}/{}".format(str(i + 1),epoch))
    tloop = tqdm(train_dataloader,leave=True)
    running_loss = 0
    model.train()
    train_acc = 0
    counter = 0

    for idx,(img,label) in enumerate(tloop):
        img=img.to(device,dtype=torch.float32)
        label=label.to(device,dtype=torch.long)
        optimizer.zero_grad()
        pred = model(img)
        _,pred_label = torch.max(pred,1)
        acc = torch.sum(pred_label==label.data)/Batchsize
        loss = BCE_loss(pred,label)
        loss.backward()
        optimizer.step()
        running_loss += loss
        train_acc += acc
        counter += 1
    print("train_loss:{}".format(running_loss.cpu().detach().numpy()) + " train_acc:{}".format(train_acc.cpu().numpy()/counter))
    train_loss_list.append(running_loss.cpu().detach().numpy())
    train_acc_list.append(train_acc.cpu().numpy()/counter)

    with torch.no_grad():
        val_loss = 0
        running_acc = 0
        model.eval()
        counter = 0

        for idx,(img,label) in enumerate(val_dataloader):
            img=img.to(device,dtype=torch.float32)
            label=label.to(device,dtype=torch.long)
            optimizer.zero_grad()
            pred = model(img)
            _,pred_label = torch.max(pred,1)
            loss = BCE_loss(pred,label)
            val_loss += loss
            acc = torch.sum(pred_label==label.data)/Batchsize
            #print(pred_label,label.data)
            running_acc += acc
            counter += 1
        print("val_loss: {}".format(val_loss.cpu().numpy()) + " running_acc: {}".format(running_acc.cpu().numpy()/counter))
        val_loss_list.append(val_loss.cpu().numpy())
        val_acc_list.append(running_acc.cpu().numpy()/counter)

print("val_loss:" + str(min(val_loss_list)))
print("val_acc:" + str(max(val_acc_list)))
print("train_loss:" + str(min(train_loss_list)))
print("train_acc:" + str(max(train_acc_list)))
fig,ax = plt.subplots(1,2)
ax[0].plot(range(epoch),train_loss_list)
ax[0].plot(range(epoch),val_loss_list)
ax[1].plot(range(epoch),train_acc_list)
ax[1].plot(range(epoch),val_acc_list)
plt.show()

#%%
torch.save(model.state_dict(),"resnet50.pth")

