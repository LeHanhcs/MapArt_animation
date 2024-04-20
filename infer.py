import os
import argparse
import torch
import datetime
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import cv2, time
import numpy as np
from network import ReCoNet
from utilities import *
import re
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='video', required=True, help='video')
parser.add_argument('--source', required=True, help='Video file with map')
parser.add_argument('--model', required=True, help='Model state_dict file')
parser.add_argument('--outputpath', type=str, default="./output/", help='Output filename')
args = parser.parse_args()


def load_test_image(filename, size=None, scale=16):
    img = Image.open(filename).convert('RGB')
  
    img = img.resize((int(img.size[0] / scale * scale), int(img.size[1] / scale* scale)), Image.ANTIALIAS)
    return img


def save_testimage(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


device = 'cuda'

content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

with torch.no_grad():
    style_model = ReCoNet()
    state_dict = torch.load(args.model)
   
    style_model.load_state_dict(state_dict)
    style_model.to(device)

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    frame_list = os.listdir(args.source)
    frame_list.sort()
    for frame in frame_list:
        start = time.time()
        content = load_test_image(args.source+"/"+frame, 320)
        content = content_transform(content)
        content = content.unsqueeze(0).to(device)       
        output = style_model(content)
        output = output.squeeze(0).cpu()
        fname = args.outputpath+ frame
        print(time.time()-start)
        print(fname)
        save_testimage(fname, output)
        
   