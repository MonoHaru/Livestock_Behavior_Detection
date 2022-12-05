import torch

layer_viz = outputs[0][0][1]
layer_viz.shape
torch.Size([224, 224])
layer_viz = layer_viz.data
print(enumerate(layer_viz))
<enumerate object at 0x0000019CA73C8C40>
enumerate(layer_viz)
<enumerate object at 0x0000019CFBD35B40>
for i, filter in enumerate(layer_viz):
    print(i)
    print(filter)

for i, filter in enumerate(layer_viz):
    if i == 16:
        break
    plt.subplot(2, 8, i + 1)
    plt.imshow(outputs[0][0][1], cmap='gray')
    plt.axis("off")
plt.show()
plt.close()


outputs[0][0][1] = torch.reshape([-1, 224, 224])

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model
modelVGG = models.vgg16(pretrained=True)
print(modelVGG)

#이미지->픽셀 데이터의 NumPy 배열로 변환, 3D->4D배열 [samples, rows, cols, channels]
#bgr->rgb

img = cv.imread("C:/Users/default.DESKTOP-A0V01KV/KakaoTalk_20220126_155434635.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img) #,interpolation='bicubic'
plt.show()

# img_B = cv.imread("D:/mat/dream8_background.PNG")
# img_B = cv.cvtColor(img_B, cv.COLOR_BGR2RGB)
# plt.imshow(img_B) #,interpolation='bicubic'
# plt.show()

#
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#0.485, 0.456, 0.406
])

img = np.array(img)
img = transform(img)
img = img.unsqueeze(0)
print(img.size()) #torch.size([1,3,224,224]

#conv layer
no_of_layers = 0# layer 초기화
conv_layers = []


#model 중간 특징 추출하기
model_children = list(modelVGG.children())

for child in model_children:
    if type(child) == nn.Conv2d:
        no_of_layers += 1
        conv_layers.append(child)

    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)
print(no_of_layers) #13


results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results

# [첫번쪠 레이어][ ][첫번쩨채널] 사이즈: 224,224
output = outputs[0][0][0].data()

layer_viz = output

plt.subplot(2, 8, i + 1)
plt.imshow(filter,cmap='gray')
plt.axis("off")
plt.show()
plt.close()

#
# # 피쳐맵 시각화
# for num_layer in range(len(outputs)):
#     plt.figure(figsize=(50, 10))
#     layer_viz = outputs[num_layer][0, :, :, :]
#     layer_viz = layer_viz.data
#     print("Layer ",num_layer+1)
#     #print(layer_viz)
#     for i, filter in enumerate(layer_viz):
#         if i == 16:
#             break
#         plt.subplot(2, 8, i + 1)
#         plt.imshow(filter,cmap='gray')
#         plt.axis("off")
#     plt.show()
#     plt.close()