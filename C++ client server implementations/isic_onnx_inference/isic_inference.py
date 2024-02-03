# importing onnx after torch may cause segmentation faults
import torch
import torchvision.transforms as transforms
import numpy as np
import onnxruntime
from PIL import Image
import sys

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_filename = sys.argv[1]

mean = [166.43850410293402 / 255.0, 133.5872994553671 / 255.0, 132.33856917079888 / 255.0]
std =  [59.679343313897874 / 255.0, 53.83690126788451 / 255.0, 56.618447349633676 / 255.0]

val_transform = transforms.Compose([  
    transforms.Resize((224,224)),                                                   
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)                                              
]) 

image = Image.open(img_filename)
image = val_transform(image)
torch_input = image.unsqueeze(0) 

# export model to onnx
onnx_name = 'model0.onnx'
ort_session = onnxruntime.InferenceSession(onnx_name)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_input)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0])
