#%%
from albumentations.augmentations.transforms import RandomBrightnessContrast
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from torch.utils.data import Dataset,DataLoader, dataloader
from torch.utils.data.dataset import Subset
import pathlib
import random
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from pprint import pprint

#%%
model_name = 'weight/resnet50_730.pth'
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

class Mydataset(Dataset):
    def __init__(self,dataset_path):
        d_path = pathlib.Path(dataset_path)
        self.img_path_list = list(d_path.glob("*.tif"))
        random.shuffle(self.img_path_list)
        self.transforms = A.Compose([
			A.Resize(256,340),
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
        print(stem)
        label = float(self.label_list.index(stem))
        augmentated = self.transforms(image=img)["image"]
        img_path = self.img_path_list[idx]

        return augmentated , torch.tensor(label),str(img_path)



device = torch.device("cuda")
model.to(device)

model.load_state_dict(torch.load(model_name))
import torch.nn.functional as F
import matplotlib.cm as cm

class _BaseWrapper(object):
  def __init__(self, model):
    super(_BaseWrapper, self).__init__()
    self.device = next(model.parameters()).device
    self.model = model
    self.handlers = []

  def _encode_one_hot(self, ids):
    one_hot = torch.zeros_like(self.logits).to(self.device)
    one_hot.scatter_(1, ids, 1.0)
    return one_hot

  def forward(self, image):
    self.image_shape = image.shape[2:]
    self.logits = self.model(image)
    self.probs = F.softmax(self.logits, dim=1)
    return self.probs.sort(dim=1, descending=True)

  def backward(self, ids):
    one_hot = self._encode_one_hot(ids)
    self.model.zero_grad()
    self.logits.backward(gradient=one_hot, retain_graph=True)


class GradCAM(_BaseWrapper):
  def __init__(self, model, candidate_layers=None):
    super(GradCAM, self).__init__(model)
    self.fmap_pool = {}
    self.grad_pool = {}
    self.candidate_layers = candidate_layers

    def _save_fmaps(key):
      def forward_hook(module, input, output):
        self.fmap_pool[key] = output.detach()
      return forward_hook

    def _save_grads(key):
      def backward_hook(module, grad_in, grad_out):
        self.grad_pool[key] = grad_out[0].detach()
      return backward_hook

    for name, module in self.model.layer4[2].named_modules():
      """
      ここのfor文で、self.modelの各層の出力を取得できるように、hookを定義
      """
      if self.candidate_layers is None or name in self.candidate_layers:
        self.handlers.append(module.register_forward_hook(_save_fmaps(name)))
        self.handlers.append(module.register_backward_hook(_save_grads(name)))


  def _find(self, pool, target_layer):
    """
    Return a map of a specific layer
      pool: map of output layer (fmap_pool or grad_pool)
      target_layer: name of layer in self.model
    """

    if target_layer in pool.keys():
      return pool[target_layer]
    else:
      raise ValueError("Invalid layer name: {}".format(target_layer))


  def generate(self, target_layer):

    fmaps = self._find(self.fmap_pool, target_layer)
    gradients = self._find(self.grad_pool, target_layer)
    weights = F.adaptive_avg_pool2d(gradients, 1)

    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam = F.relu(gcam)
    gcam = F.interpolate(
        gcam, self.image_shape, mode='bilinear', align_corners=False
    )

    B, C, H, W = gcam.shape

    # おそらく正規化
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= gcam.max(dim=1, keepdim=True)[0]
    gcam = gcam.view(B, C, H, W)

    return gcam

def out_gradcam(gcam, raw_image, paper_cmap=False):
  gcam = gcam.cpu().numpy()
  cmap = cm.jet(gcam)[...,:3] * 255.0

  if paper_cmap:
    alpha = gcam[..., None]
    gcam = alpha * cmap + (1 - alpha) * raw_image
  else:
    gcam = (cmap.astype(np.float64) + raw_image.astype(np.float64)) / 2

  return np.uint8(gcam)
#%%
import os 
import csv
header = ['path','pred','label']
data_name='ca'
name = '_ca2'
dataset = Mydataset('grad_val/{}'.format(data_name))
print(len(dataset))
g_dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
body = []
for (data,label,path) in g_dataloader:
    #torch.cuda.empty_cache()

    input_data = data.to(device,dtype=torch.float32)
    raw_image = input_data[0].to('cpu').detach().numpy().copy()
    raw_image = raw_image.transpose((1, 2, 0))
#    raw_image = ((raw_image * 0.5) + 0.5) * 255.0
    raw_image = raw_image.astype(np.uint8)

    output = nn.Softmax(dim=1)(model(input_data))
    conf,pred = torch.max(output.data,1)
    print(
    "Output: {} \n Confidence: {} \n Predicted: {} \n Answer: {}".format(output, conf, pred, label)
)
    body.append([path,pred,label])
    
    gcam = GradCAM(model=model)
    _ = gcam.forward(input_data)
    single_predicted = pred.view(1, pred.shape[0]).to(device)
    gcam.backward(ids=single_predicted)
    regions = gcam.generate("conv3")
    output = out_gradcam(gcam=regions[0, 0],raw_image=raw_image)
    print(output.shape)
    path = pathlib.Path(path[0])
    num = int(path.stem)
    (x,y) = (num%4,num//4)
    print(x,y)
    output = Image.fromarray(output)
    output.save("./grad_val/result/grad{}/".format(name)+str(num)+".tif")
#%%
with open('grad_val/result/new_csv/{}.csv'.format(name),'w') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  writer.writerows(body)

f.close()
