import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2

device = 'cuda'
IMG_SIZE = (320, 512)#(320, 320)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# To tensor will change the value to 0-1
transform1 = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))])
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
mask_transform = transforms.Compose([
			transforms.Resize(IMG_SIZE),
	        transforms.ToTensor()])
style_transform = transforms.Compose([
			transforms.Resize(IMG_SIZE),
	        transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))])

crop_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(128),
	        transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))])

def normalizeVGG19(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch/255.0
    batch = batch.div_(255.0)
    return (batch - mean) / std
def gram_matrix(feature_map):
    n, c, h, w = feature_map.shape
    feature_map = feature_map.reshape((n, c, h * w))
    return feature_map.bmm(feature_map.transpose(1, 2)) / (c * h * w)
def gram_matrix_mul(feature_map_1, feature_map_2):
    if feature_map_1.shape != feature_map_2.shape:
        feature_map_2 = torch.nn.functional.interpolate(feature_map_2, scale_factor=2)
    n, c, h, w = feature_map_2.shape
    feature_map_2 = feature_map_2.reshape((n, c, h * w))
    n, c, h, w = feature_map_1.shape
    feature_map_1 = feature_map_1.reshape((n, c, h * w))
    return feature_map_1.bmm(feature_map_2.transpose(1, 2)) / (c * h * w)

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img
def l2_squared(x):
    return x.pow(2).sum()
def rgb_to_luminance(x):
    return x[:, 0, ...] * 0.2126 + x[:, 1, ...] * 0.7512 + x[:, 2, ...] * 0.0722

def save_image(data, name, flag):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    if flag:
        img.save('./trainout/single/{}.jpg'.format(name))
    else:
        img.save('./trainout/video/{}.jpg'.format(name))
def savetest(data, model, epoch, itr):
    with torch.no_grad():
        test = model(data[0].to(device))
        test = test.cpu()
        save_image(test[0], str(epoch)+"_"+str(itr), 1)
        test1 = model(data[1].to(device))
        test1 = test1.cpu()
        save_image(test1[0], str(epoch)+"_"+str(itr)+"_1", 0)
        test2 = model(data[2].to(device))
        test2 = test2.cpu()
        save_image(test2[0], str(epoch)+"_"+str(itr)+"_2", 0)

