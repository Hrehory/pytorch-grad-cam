#!/usr/bin/env python
# -*- coding: utf-8 -*-

# project modules
from io_utils import preprocess_image

# torch modules
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import VGG

# science modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

# misc
import sys






'''
def backward(gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
    print (gradient)
    torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)
'''

model = torchvision.models.vgg19(pretrained=True)
#model.backward = backward


def guided_hook(grad):
    print (grad.size())
    grad[grad < 0] = 0.0
    return grad

for name, param in model.named_parameters():
    param.register_hook(guided_hook) 


img = cv2.imread(sys.argv[1], 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
input = preprocess_image(img)
output = model(input)


values, indices = torch.max(output, 0)
winning_class = np.argmax(values.data.numpy())
target = Variable(torch.zeros(1000))
target[winning_class] = 1


criterion = nn.MSELoss()

loss = criterion(output, target)
loss.backward()

#for name, param in model.named_parameters():
#    print (param.grad)

gradient_img = input.grad.data.numpy().reshape(224,224,3)
gray = cv2.cvtColor(gradient_img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')

plt.show()
