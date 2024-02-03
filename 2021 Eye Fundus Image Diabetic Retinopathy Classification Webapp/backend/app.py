import math
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json

import torch
import numpy as np
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import cv2

from PIL import Image, ImageDraw

import requests

import base64
from io import BytesIO


import os
import sys
import logging
import logging.handlers

class CenteredCircularMaskTensor(object):
    def __init__(self, size, val_mask = []):
        img_mask = Image.new('1', (size,size))
        draw = ImageDraw.Draw(img_mask)
        draw.ellipse((0,0,size,size), fill = 'white', outline ='white')
        mask = np.asarray(img_mask).astype('float32')
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).expand(3,size,size)
        self.mask = mask
        self.summask = False
                
        if len(val_mask) > 0:
            self.summask = True
            mask_inverted = torch.from_numpy(np.logical_not(np.asarray(img_mask).astype('bool_')).astype('float32'))
            self.mask_sum = torch.zeros(3,size,size)
            for i in range(len(val_mask)):
                self.mask_sum[i] += val_mask[i]*mask_inverted
        
    def __call__(self,tensor):
        val = torch.mul(tensor, self.mask)
        if self.summask:
            val = val + self.mask_sum
        return val 

class RetineNet(nn.Module):
    def __init__(self, features):
        super(RetineNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),            
            nn.AvgPool2d(kernel_size=4),
        ) 
        self.output = nn.Sequential(nn.Linear(64, 4))
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)        
        return x

def make_layers(cfg, batch_norm=False, conv_size=3, pad_size=1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_size, padding=pad_size)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def ret6_bn_pretrained(arch_path = "model_best_husjr_val_0.929.tar", **kwargs):
    cfg = {
        'F': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M']    
    }
    device = 'cpu'
    model = RetineNet(make_layers(cfg['F'], batch_norm=True, conv_size=3, pad_size=1), **kwargs)
    checkpoint = torch.load(arch_path, map_location=torch.device(device))    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model

log = logging.getLogger(__name__)
handler = logging.FileHandler("backend.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)
sys.stderr.write = log.error 
sys.stdout.write = log.info 


# Load prediction related variables
mean = [0.382, 0.265, 0.189]
std = [0.290, 0.210, 0.170]
normvect = [-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]]
normalize = transforms.Normalize(mean=mean, std=std)   
    
preprocess = transforms.Compose([
    transforms.Resize(640),
    transforms.ToTensor(),
    normalize,
    CenteredCircularMaskTensor(640, normvect)
])

# load model
architecture = ret6_bn_pretrained()
architecture.eval()

# All classes
class_labels = 'dr_classes.json'

# Read the json
with open('dr_classes.json', 'r') as fr:
	json_classes = json.loads(fr.read())

app = Flask(__name__)

# Allow 
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route("/")
#def hello():
#	return "Hello World!"

def PILimage_to_base64_utf8(img):
    #img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		#print("request data", request.data)
		#print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			
			# Send uploaded image for prediction
			json_prediction_data = predict_img(UPLOAD_FOLDER+filename)
			#print("predicted_image_class", predicted_image_class)

	return json_prediction_data

def center_retina(im, tol=50):
    szx,szy, _ = im.shape
    cx = float(im.shape[0])*0.5
    cy = float(im.shape[1])*0.5
    
    minval = float(min(szx,szy))
    stepx = float(szx) / minval
    stepy = float(szy) / minval
    
    posx = 0.0
    posy = 0.0
    while posx < szx/2:
        if np.sum(im[int(posx), int(posy), :]) > tol:
            break
        posx += stepx
        posy += stepy
    D1 = 2*math.sqrt((cx-posx)**2 + (cy-posy)**2)
    
    posx = szx-1
    posy = 0.0
    while posx > szx/2:
        if np.sum(im[int(posx), int(posy), :]) > tol:
            break
        posx -= stepx
        posy += stepy
    D2 = 2*math.sqrt((cx-posx)**2 + (cy-posy)**2)
    
    posx = 0.0
    posy = szy-1
    while posx < szx/2:
        if np.sum(im[int(posx), int(posy), :]) > tol:
            break
        posx += stepx
        posy -= stepy
    D3 = 2*math.sqrt((cx-posx)**2 + (cy-posy)**2)
    
    posx = szx-1
    posy = szy-1
    while posx < szx/2:
        if np.sum(im[int(posx), int(posy), :]) > tol:
            break
        posx -= stepx
        posy -= stepy
    D4 = 2*math.sqrt((cx-posx)**2 + (cy-posy)**2)

    max_val = max(D1,D2,D3,D4)
    min_val = min(D1,D2,D3,D4)
    candidate1 = (D1+D2+D3+D4-max_val)/3.0
    candidate2 = (D1+D2+D3+D4-min_val)/3.0
    dist_max = max_val - candidate1
    dist_min = min_val - candidate2
    if dist_max > dist_min:
        D = int(candidate1)
    else: 
        D = int(candidate2)
    
    #print(f"Diameter={D} szx={szx} szy={szy}")
        
    if D > szx:
        bordersize = int((D-szx)/2)
        im=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    if D > szy:
        bordersize = int((D-szy)/2)
        im=cv2.copyMakeBorder(im, top=0, bottom=0, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    
    x = int((szx-D)/2)
    y = int((szy-D)/2)
    if D < szx and D < szy:
        # crop in both directions
        im = im[x:x+D, y:y+D, :]
    elif D < szx:
        im = im[x:x+D, :, :]
    elif D < szy:
        im = im[:, y:y+D, :]
        
    im = cv2.resize(im, dsize=(640, 640), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(im)

def predict_img(img_path):
	print(f"Entering predict_img function")

	# Path to uploaded image
	path_img = img_path

	print(f"Reading image {path_img}")
	# Read uploaded image
	read_img = Image.open(path_img)
        
	# Convert image to RGB if it is a .png
	if path_img.endswith('.png'):
	    read_img = read_img.convert('RGB')

	print(f"Centering retina")
	# center retina
	read_img = center_retina(np.array(read_img))

	with torch.no_grad():
	    print(f"Applying preprocessing")
	    img_tensor = preprocess(read_img)
	    img_tensor.unsqueeze_(0)
	    img_variable = Variable(img_tensor)
	    print(f"Predicting")
	    # Predict the image
	    outputs = architecture(img_variable)
	    probs = torch.softmax(outputs, axis=1).detach().numpy()[0]

	print(f"{probs[0]:.2f} {probs[1]:.2f} {probs[2]:.2f} {probs[3]:.2f}")
	label = json_classes[str(probs.argmax())]
	#print("\n Answer: ", label)

	out = {
		"label": label,
		"c0": int(probs[0]*100 + 0.5),
		"c1": int(probs[1]*100 + 0.5),
		"c2": int(probs[2]*100 + 0.5),
		"c3": int(probs[3]*100 + 0.5),
		"img": PILimage_to_base64_utf8(read_img)
	}

	return json.dumps(out)


if __name__ == "__main__":
	app.run('0.0.0.0', 5000, debug=True)
