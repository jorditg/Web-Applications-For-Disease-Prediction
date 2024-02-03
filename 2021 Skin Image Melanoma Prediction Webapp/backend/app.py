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

from efficientnet_pytorch import EfficientNet

import os
import sys
import logging
import logging.handlers

#log = logging.getLogger(__name__)

#handler = logging.FileHandler("backend.log")
#formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#handler.setFormatter(formatter)
#log.addHandler(handler)
#sys.stderr.write = log.error 
#sys.stdout.write = log.info 

def load_model_efficientnetB5(arch):
	print("loading load_model_efficientnetB5")
	checkpoint = torch.load(arch,  map_location={'cuda:0': 'cpu'})
	model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
	model.load_state_dict(checkpoint['state_dict'])
	print("model loaded")
	return model


input_size = 456
checkpoint = "checkpoint_0_20_9_0.953.pth.tar"
# All classes
class_labels = 'dr_classes.json'

# mean and stdev from ISIC dataset
mean = [166.43850410293402 / 255.0, 133.5872994553671 / 255.0, 132.33856917079888 / 255.0]
std =  [59.679343313897874 / 255.0, 53.83690126788451 / 255.0, 56.618447349633676 / 255.0]   

normalize = transforms.Normalize(mean=mean, std=std)
    
# Preprocessing according to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Example seen at https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

preprocess = transforms.Compose([
	transforms.Resize(input_size),
	transforms.ToTensor(),
	normalize
])

# Choose which model achrictecture to use from list above
architecture = load_model_efficientnetB5(checkpoint)
architecture.eval()

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
		print("POST")
		#print("request data", request.data)
		#print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']
		data = {
			'file': request.files['file'].filename,
			'crop_posx': request.form['crop_posx'],
			'crop_posy': request.form['crop_posy'],
			'crop_posw': request.form['crop_posw'],
			'crop_posh': request.form['crop_posh'],
			'orig_width': request.form['orig_width'],
			'orig_height': request.form['orig_height'],
			'operations': request.form['operations'],
			"label": 0,
			"normal": 0,
			"melanoma": 0,
			"img": "",
			"model": 'EfficientNet-B5'			
		}

		#print(data)

		#print(file.filename)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			
			# Send uploaded image for prediction
			json_prediction_data = predict_img(UPLOAD_FOLDER+filename, architecture, data)
			#print("predicted_image_class", predicted_image_class)

	return json_prediction_data

	
def predict_img(img_path, architecture, data):
	print(f"Entering predict_img function")       

	# Path to uploaded image
	path_img = img_path

	print(f"Reading image {path_img}")
	# Read uploaded image
	read_img = Image.open(path_img)
        
	# Convert image to RGB if it is a .png
	if path_img.endswith('.png'):
		read_img = read_img.convert('RGB')
            
	#print(f"Applying preprocessing")
	#img_tensor = preprocess(read_img)
	#img_tensor.unsqueeze_(0)
	#img_variable = Variable(img_tensor)

	print(f"Predicting")
	# Predict the image
	#outputs = architecture(img_variable)
	
	with torch.no_grad():
		outputs = architecture(Variable(preprocess(read_img).unsqueeze_(0)))
		pred = torch.sigmoid(outputs).data.cpu().numpy()[0]

	#print(pred.shape)

	print(f"Normal: {(1.0-pred[0]):.2f} Melanoma: {pred[0]:.2f}")
	
	if pred[0] > 0.5:
		label = 1
	else:
		label = 0

	label = json_classes[str(label)]
	#print("\n Answer: ", label)

	# out = {
	# 	"label": label,
	# 	"normal": int((1.0-pred[0])*100 + 0.5),
	# 	"melanoma": int(pred[0]*100 + 0.5)
	# 	#"img": PILimage_to_base64_utf8(read_img),
	# 	'model': 'EfficientNet-B5'
	# }

	data["label"] = label
	data["normal"] = int((1.0-pred[0])*100 + 0.5)
	data["melanoma"] = int(pred[0]*100 + 0.5)
	#data["img"] = PILimage_to_base64_utf8(read_img)

	json_object = json.dumps(data)
	
	# Writing to sample.json
	with open(f"{os.path.splitext(path_img)[0]}.json", "w") as outfile:
		outfile.write(json_object)

	return json_object

if __name__ == "__main__":
	app.run('0.0.0.0', 5001, debug=True)
