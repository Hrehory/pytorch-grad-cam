#!/usr/bin/env python
# -*- coding: utf-8 -*-

# project modules
from io_utils import preprocess_image

# torch modules
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn


# science modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

# misc
import sys


model = torchvision.models.vgg19(pretrained=True)
#for param in model.named_parameters():
#    print (param)


img = cv2.imread(sys.argv[1], 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
input = preprocess_image(img)
output = model(input)


values, indices = torch.max(output, 0)
winning_class = np.argmax(values.data.numpy())
target = Variable(torch.zeros(1000))
target[winning_class] = 1


criterion = nn.MSELoss()

loss = criterion(output.clamp(0,1), target)
loss.backward()

gradient_img = input.grad.data.numpy().reshape(224,224,3)
gray = cv2.cvtColor(gradient_img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')

plt.show()
