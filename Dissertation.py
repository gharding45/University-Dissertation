# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:26:48 2025

@author: George Harding
"""

import requests
import random
import urllib.request
from PIL import Image
import re
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel, BertTokenizer, BertModel, AutoProcessor, Blip2ForConditionalGeneration
import easyocr
from geopy.geocoders import Nominatim
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle #saving file
from datetime import datetime
import cv2
import copy
import numpy as np

'''
Python Library References:
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q. and Rush, A. 2020. Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. [Online]. [Accessed 6 February 2025]. Available from: https://doi.org/10.18653/v1/2020.emnlp-demos.6
- NVIDIA Corporation. 2022. NVIDIA CUDA Toolkit (version 11.8.90). [Software]. [Accessed 14 February 2025]. Available from: https://developer.nvidia.com/cuda-toolkit
- Jaided AI. 2024. EasyOCR (version 1.7.2). [Software]. [Accessed 8 February 2025]. Available from: https://github.com/JaidedAI/EasyOCR
- OpenCV. 2025. OpenCV (version 4.11.0). [Software]. [Accessed 1 March 2025]. Available from: https://opencv.org/
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32. [Online]. pp8024-8035. [Accessed 6 February 2025]. Available from: https://doi.org/10.48550/arXiv.1912.01703
- Geopy. 2025. Geopy (version 2.4.1). [Software]. [Accessed 8 February 2025]. Available from: https://github.com/geopy/geopy
- Numpy. 2024. Numpy (version 1.26.4). [Software]. [Accessed 19 January 2025]. Available from: https://numpy.org/
- Pillow. 2024. Pillow (version 10.3.0). [Software]. [Accessed 8 February 2025]. Available from: https://python-pillow.github.io/
- Requests. 2024. Requests (version 2.32.2).[Software]. [Accessed 8 February 2025]. Available from: https://requests.readthedocs.io/en/latest/
'''

#Ensure GPU is utilised for efficient processing
device = "cuda" if torch.cuda.is_available() else "cpu"

#Initialise geocell global variables:
    
#BBox1 captures most of uk and ireland
longMin1 = -11.13935
longMax1 = 1.77517
latMin1 = 50.96091
latMax1 = 60.91659

##BBox2 captures south uk, cornwall. Avoids france. Is about 5-10% the size as bbox1
longMin2 = -6.54309
longMax2 = 0.98602
latMin2 = 49.88175
latMax2 = 50.96228

#Overal BBox, full range we are working with for geocell calculations
longMin = min(longMin1, longMin2) # -11.13935
longMax = max(longMax1, longMax2) #1.77517
latMin = min(latMin1, latMin2) #49.88175
latMax = max(latMax1, latMax2) #60.91659

#set geocell size for model
geocell_size = 0.098 #each cell is geocell_size x geocell_size degrees in size.
lat_cell_count = int((latMax - latMin) // geocell_size)
long_cell_count = int((longMax - longMin) // geocell_size)
total_geocells = lat_cell_count * long_cell_count



''' Function setGeocellSize():
Use:
Updates the global geocell variables for switching between models which use different geocell sizes.
'''
def setGeocellSize(size): 
    global geocell_size, lat_cell_count, long_cell_count, total_geocells  #access global variables
    geocell_size = size #size x size degrees cells
    lat_cell_count = int((latMax - latMin) // geocell_size)
    long_cell_count = int((longMax - longMin) // geocell_size)
    total_geocells = lat_cell_count * long_cell_count



''' Function coordinates_to_geocell():
Use:
Converts the UK into geocells, based on the global bbox coordinates. Designed for batches.
geocell_size is set as a global variable. If geocell_size is 1, eaach geocell will be 1 degree x 1 degree in size

Inputs/outputs:
target_coordinates: a coordinate (allows batches)
returns: the geocell the coordinate is in (allows batches)
'''
def coordinates_to_geocell(target_coordinates):
    #mapillary coordinates are (longitude, latitude) format
    long, lat = target_coordinates[:, 0], target_coordinates[:, 1]

    #latMin, longMin, latMax, longMax, long_cell_count are defined as global variables
    latId = (lat - latMin) // geocell_size
    longId = (long - longMin) // geocell_size

    #for batches, use .long() instead of int()
    geocell_list = (latId.long() * long_cell_count + longId.long()).long() #convert batch to 1d (flatten)
    return geocell_list



''' Function geocell_to_coordinate():
Use:
Gets the coordinates of the centre of a specified geocell. Reverses the coordinates_to_geocell() function. Designed for batches.

Inputs/outputs:
geocell: the geocell to find the coordinates of (allows batches)
returns: coordinates of the geocell (allows batches)
'''
def geocell_to_coordinate(geocell):
    #latMin, longMin, latMax, longMax, long_cell_count are defined as global variables
    latId = geocell // long_cell_count 
    longId = geocell % long_cell_count #remainder is longId
    
    lat = latId * geocell_size +latMin
    long = longId * geocell_size + longMin
    centre = geocell_size / 2 #offset to the centre
    
    lat = lat.clone().detach().float()
    long = long.clone().detach().float()
    
    return torch.stack([long+centre, lat+centre], dim=1) #return the centre of the geocell



''' Function getData():
Use:
fetches street view images from Mapillary's public dataset using Python's urllib request library. 

Inputs/outputs:
returns: an array of data items

References:
- Mapillary. 2025. Mapillary API Documentation. [Online]. [Accessed 3 February 2025]. Available from: https://www.mapillary.com/developer/api-documentation
'''
def getData():
    endpoint = "https://graph.mapillary.com/images"
    accessToken = "MLY|9137912079588378|53aa91e9cb13c1285b46b0a97feea1e7" #personal token from the API website
    fields = "id,computed_geometry,thumb_2048_url" #fields to fetch
    
    #BBox1 captures most of uk and ireland
    bbox1 = f"{longMin1},{latMin1},{longMax1},{latMax1}"
    #BBox2 captures south uk, cornwall. Avoids france. Is about 5-10% the size as bbox1
    bbox2 = f"{longMin2},{latMin2},{longMax2},{latMax2}"
        
    #parameters for api request
    parameters1 = {
        'access_token': accessToken,
        'fields': fields,
        'bbox': bbox1
    }
    
    parameters2 = {
        'access_token': accessToken,
        'fields': fields,
        'bbox': bbox2
    }
    
    response1 = requests.get(endpoint, params=parameters1)
    if response1.status_code == 200:
        data1 = response1.json() 
    else:
        data1 = None
        print(f"Error: {response1.status_code}, {response1.text}")
    
    response2 = requests.get(endpoint, params=parameters2)
    if response2.status_code == 200:
        data2 = response2.json() 
    else:
        data2 = None
        print(f"Error: {response2.status_code}, {response2.text}")

    #combine data:
    data = data1.get('data', []) + data2.get('data', [])[:int(len(data2.get('data', [])) * 0.1)] #scale bbox2 to be proportional (about 10% the size)
    random.shuffle(data) #randomise order
    return data



''' Function getCaption():
Use:
Prompted image captioning. Generates a caption for an image using the model BLIP2, which is pretrained for image to text zero shotting.

Inputs/outputs:
model: a blip2-opt-2.7b model
processor: a blip2-opt-2.7b processor
img: an image to generate a caption for
returns: the caption

References:
- Li, J., Li, D., Savarese, S. and Hoi, S. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML. [Online]. [Accessed 6 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2301.12597
- Khalusova, M. and Li, J. 2023. Zero-shot image-to-text generation with BLIP-2. 15 February. Hugging Face. [Online]. [Accessed 6 February 2025]. Available from: https://huggingface.co/blog/blip-2
'''
def getCaption(model, processor, img):    
    prompt = """Question: Describe this UK location concisely, including road type, buildings, signs, vegetation, and surroundings. Answer:"""
    inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=30, # allow more words for detail
                                       do_sample=False, # Make it deterministic
                                       num_beams=3,
                                       repetition_penalty=1.5)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #generated_text is of the form "Question: {} Answer: {}"
    caption = generated_text.split("Answer:")[-1].strip()
    return caption



''' Function getTextEncoding():
Use:
Generates a text encoding of the caption using the model BERT, which is pretrained and able to generate meaningful text embeddings.

Inputs/outputs:
model: a bert-base-uncased model
tokenizer: a bert-base-uncased tokenizer
text: the text to generate text embeddings for
returns: text embedding

References:
- Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint. [Online]. [Accessed 8 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.1810.04805
- GeeksforGeeks. 2024. How to Generate Word Embedding using BERT? [Online]. [Accessed 8 February 2025]. Available from: https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
- Alammar, J. 2019. A Visual Guide to Using BERT for the First Time. 26 November. Visualizing machine learning one concept at a time. [Online]. [Accessed 12 February 2025]. Available from: https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
'''
def getTextEncoding(model, tokenizer, text):
    encoding = tokenizer.batch_encode_plus( [text],
        padding=True, 
        truncation=True, 
        return_tensors='pt', # PyTorch tensors
        add_special_tokens=True # Add CLS token
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)   
    with torch.no_grad(): # No gradient computation
        outputs = model(input_ids, attention_mask=attention_mask)
        CLS_token_text = outputs.last_hidden_state[:, 0, :] #get CLS token
    
    return CLS_token_text



''' Function getImageEncoding():
Use:
Generates a visual encoding of the image using the model ViT, which is pretrained and able to generate meaningful visual embeddings.

Inputs/outputs:
model: a vit-base-patch16-224-in21k model
processor: a vit-base-patch16-224-in21k processor
img: the image to generate visual embeddings for
returns: visual embedding

References:
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. [Online]. [Accessed 12 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2010.11929
- Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. 2009 IEEE conference on computer vision and pattern recognition. pp.248-255.
'''
def getImageEncoding(model, processor, img):
    inputs = processor(img, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad(): # No gradient computation
        outputs = model(**inputs)
        CLS_token_image = outputs.last_hidden_state[:, 0, :] #get CLS token
        
    return CLS_token_image



''' Function crossAttention():
Use:
Generate a fused encoding by doing cross attention on the textual and visual encodings.

Inputs/outputs:
img_encoding: the image encoding generated by getImageEncoding()
text_encoding: the text encoding generated by getTextEncoding()

References:
- xbeat. 2024. Self-Attention and Cross-Attention in Transformers with Python. [Online]. [Accessed 13 February 2025]. Available from: https://github.com/xbeat/Machine-Learning/blob/main/Self-Attention%20and%20Cross-Attention%20in%20Transformers%20with%20Python.md
'''
def crossAttention(img_encoding, text_encoding):
    scores = torch.matmul(text_encoding, img_encoding.transpose(-2, -1)) / (text_encoding.size(-1) ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, img_encoding)



''' Class HaversineLoss():
Use:
Calculates the average haversine distance of a batch, used as the loss for training models with the PyTorch library.

Inputs/outputs:
pred: predicted coordiantes generated by the model (lat1 and long1)
target: actual coordinates (lat2 and long2)
returns: the mean distance between the predictions and actual coordinates to be used as the loss. 

References:
- Oppidi, A. and Jha, A. 2025. PyTorch Loss Functions: The Ultimate Guide. 27 January. Neptune Blog. [Online]. [Accessed 20 February 2025]. Available from: https://neptune.ai/blog/pytorch-loss-functions
- avitex, 2011. Haversine formula in Python (bearing and distance between two GPS points). [Online]. [Accessed 20 February 2025]. Available from: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
- Louwers, J. 2023. Calculate Geographic distances in Python with the Haversine method. 18 September. Medium. [Online]. [Accessed 20 February 2025]. Available from: https://louwersj.medium.com/calculate-geographic-distances-in-python-with-the-haversine-method-ed99b41ff04b
- DrTime. 2023. Custom loss function failure: distance between two points on globe (Haversine Loss). [Online]. [Accessed 20 February 2025]. Available from: https://discuss.pytorch.org/t/custom-loss-function-failure-distance-between-two-points-on-globe-haversine-loss/190610
'''
class HaversineLoss(nn.Module):
   def __init__(self, earthRadiusKm = 6371):
       super(HaversineLoss, self).__init__()
       self.radius = earthRadiusKm

   def forward(self, pred, target):
       # pred shape [batch_size, 2] (lat, long)
       # target shape [batch_size, 2] (lat,long)
       latPred, longPred = torch.split(pred, 1, dim=1)  
       latTarget, longTarget = torch.split(target, 1, dim=1)  
       
       #convert to radians
       latPredRad = torch.deg2rad(latPred)
       longPredRad = torch.deg2rad(longPred)
       latTargetRad = torch.deg2rad(latTarget)
       longTargetRad = torch.deg2rad(longTarget)

       #distance between lat and long values
       deltaLat = latTargetRad - latPredRad
       deltaLong = longTargetRad - longPredRad
       
       #haversine formula
       a = torch.sin(deltaLat/2)**2 + torch.cos(latPredRad) * torch.cos(latTargetRad) * torch.sin(deltaLong/2)**2
       c = 2 * torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
       distance = self.radius * c
       
       return torch.mean(distance)



''' function haversine_distance():
Use:
Calculates the haversine distance for a singular prediction and actual coordinate. Exactly the same as the HaversineLoss Class but for individual usage.

Inputs/outputs:
pred: predicted coordiante (lat1 and long1)
target: actual coordinate (lat2 and long2)
returns: the distance between the prediction and actual coordinate.
'''
def haversine_distance(pred, target, radius=6371):
    #allows singular and multiple items at once
    if target.dim()== 1:
        target = target.unsqueeze(0)
    
    latPred, longPred = torch.split(pred, 1, dim=1)
    latTarget, longTarget = torch.split(target, 1, dim=1)
    
    #convert to radians
    latPredRad = torch.deg2rad(latPred)
    longPredRad = torch.deg2rad(longPred)
    latTargetRad = torch.deg2rad(latTarget)
    longTargetRad = torch.deg2rad(longTarget)

    #distance between lat and long values
    deltaLat = latTargetRad - latPredRad
    deltaLong = longTargetRad - longPredRad
    
    a = torch.sin(deltaLat/2)**2 + torch.cos(latPredRad) * torch.cos(latTargetRad) * torch.sin(deltaLong/2)**2
    c = 2 * torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    distance = radius * c 
    
    return distance.squeeze(1)



''' Class MLP():
Use:
The neural network of my model to learn, classify and regress. Designed for batches.

Inputs/outputs:
fused_encodings: the fused encodings of visual and textual encodings
returns: geocell logits (probability distribution of each geocell) and the geocell offset from the centre of the predicted geocell.

References:
- Zhang, X. 2019. Multi-Layer Perceptron (MLP) in PyTorch. 26 December. Deep Learning Study Notes. [Online]. [Accessed 18 February 2025]. Available from: https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
- Tam, A. 2023. Building Multilayer Perceptron Models in PyTorch. 08 April. Deep Learning with PyTorch. [Online]. [Accessed 18 February 2025]. Available from: https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
- PyTorch. 2017. Training a Classifier. [Online]. [Accessed 19 March 2025]. Available from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), #hidden layer 1 
            nn.GELU(), #activation function
            nn.Linear(1024, 512), #hidden layer 2
            nn.GELU(), #activation function
            nn.Linear(512, 256), #hidden layer 2
            nn.GELU(), #activation function
        )    
        #predict the geocell
        self.classifier = nn.Linear(256, total_geocells)  #the total number of geocells is the number of classes
        
        #predict the lat and long inside the geocell
        self.regressor = nn.Linear(256, 2) 
            

    def forward(self, fused_encodings):
        #learn features with the neural network
        learned_features = self.mlp(fused_encodings)
        
        #predict the geocell
        geocell_logits = self.classifier(learned_features)
        
        #predict offset inside geocell, ranging from -0.05 to 0.05 degrees in long and lat. Scaled for geocell size.
        geocell_offset = (torch.sigmoid(self.regressor(learned_features)) - 0.5) * geocell_size
        return geocell_logits, geocell_offset



''' function trainModel():
Use:
Train a model and save it to a file. It splits the data into a training set and a validation set for evaluation, uses a weighted loss of haversine and cross entropy to learn. Trains in batches for efficiency. 

Inputs/outputs:
fused_encodings: the fused encodings generated with text and image encodings
target_coordinates: the actual coordinate of each image
modelName: the name of the model being created (for saving the file)
returns: 1 for success meaning the model was saved to a file.

References:
- Zhang, X. 2019. Multi-Layer Perceptron (MLP) in PyTorch. 26 December. Deep Learning Study Notes. [Online]. [Accessed 18 February 2025]. Available from: https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
- Tam, A. 2023. Building Multilayer Perceptron Models in PyTorch. 08 April. Deep Learning with PyTorch. [Online]. [Accessed 18 February 2025]. Available from: https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
- Sarah_W. 2020. Split dataset into two new datasets (NOT subset). [Online]. [Accessed 23 February 2025]. Available from: https://discuss.pytorch.org/t/split-dataset-into-two-new-datasets-not-subset/71328
- Thunder. 2022. Random split a PyTorch dataset of type TensorDataset. [Online]. [Accessed 23 February 2025]. Available from: https://stackoverflow.com/questions/73486641/random-split-a-pytorch-dataset-of-type-tensordataset
- Huang, Y.J. 2020. Pytorch schedule learning rate. [Online]. [Accessed 26 February 2025]. Available from: https://stackoverflow.com/questions/63108131/pytorch-schedule-learning-rate
- Yadav, A. 2024. Guide to Pytorch Learning Rate Scheduling. 28 October. Data Scientist's Diary. [Online]. [Accessed 26 February 2025]. Available from: https://medium.com/data-scientists-diary/guide-to-pytorch-learning-rate-scheduling-b5d2a42f56d4
- Mahdi_Amrollahi. 2022. Is there a way to combine classification and regression in single model? [Online]. [Accessed 19 March 2025]. Available from: https://discuss.pytorch.org/t/is-there-a-way-to-combine-classification-and-regression-in-single-model/165549
- LearnPyTorch.io. [no date]. 02. PyTorch Neural Network Classification. [Online]. [Accessed 19 March 2025]. Available from: https://www.learnpytorch.io/02_pytorch_classification/
'''
def trainModel(fused_encodings, target_coordinates, modelName):
    #into a tensor with shape [batch_size, 768]. The list items are already tensors so clone
    fused_encodings = [encoding.clone().detach().squeeze(0).to(device) for encoding in fused_encodings]
    fused_encodings = torch.stack(fused_encodings)

    #into a tensor with shape [batch_size, 2].
    target_coordinates = [torch.tensor(target, dtype=torch.float32, device=device) for target in target_coordinates]
    target_coordinates = torch.stack(target_coordinates)

    data = TensorDataset(fused_encodings, target_coordinates)
    
    #split into training and validation sets
    training_length = int(0.8 * len(data)) #80% is training set
    validation_length = len(data) - training_length
    training_data, validation_data = random_split(data, [training_length, validation_length])

    #batches of size 32
    training_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

    #create model
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) #half learning rate when it plateaus

    #initalise loss functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = HaversineLoss() 

    #training loop
    for epoch in range(70):
        total_loss = 0
        model.train()
        
        for fused_encoding, target_coordinate in training_dataloader:
            #Use model
            geocell_logits,offset = model(fused_encoding)
            
            geocell = torch.argmax(geocell_logits, dim=1) #get largest of logits
            geocell_coordinate = geocell_to_coordinate(geocell)
            prediction = geocell_coordinate + offset #prediction is geocell + offset.
            
            #get losses for current prediction
            regression_loss = regression_criterion(prediction, target_coordinate)
            classification_loss = classification_criterion(geocell_logits, coordinates_to_geocell(target_coordinate))

            #balance importance of the classification vs regression
            loss = classification_loss + 0.6 * regression_loss
            optimizer.zero_grad()
            loss.backward()
            
            #gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() #calculate loss across all batches
        
        #validation
        validation_loss = validateModel(model, classification_criterion, regression_criterion, validation_dataloader)
        scheduler.step(validation_loss) #adjust learning rate based on validation loss
        
        if epoch % 5 == 0:  #print every 5 epochs
            print(f"Epoch {epoch}: Training loss: {total_loss / len(training_dataloader)}, Validation Loss: {validation_loss}")
         
    #save model state    
    torch.save(model.state_dict(), f"{modelName}.pth")
    
    #clear cache
    del model
    torch.cuda.empty_cache()
    
    return 1



''' function validateModel():
Use: 
Evaluates model performance on a validation set. Built similarly to trainModel() but is not training. 

Inputs/outputs:
model: the current model being created by trainModel().
regression_criterion: the Haversine distance loss function.
validation_dataloader: the validation set consisting of fused encodings and target coordinates.
returns: The haversine distance loss for the validation set.
'''
def validateModel(model,regression_criterion, validation_dataloader):
    model.eval() #switch to evaluation mode instead of training mode
    haversine_loss = 0
    total_correct = 0
    total_samples = 0
    total_within_100m = 0
    total_within_1km = 0
    total_within_5km = 0
    total_within_10km = 0
    total_within_50km = 0
    total_within_100km = 0
    total_within_250km = 0
    distance_list = [] # Track all distances to get median at the end

    with torch.no_grad(): # No gradient computation
        for fused_encoding, target_coordinate in validation_dataloader:        
            #use model
            geocell_logits,offset = model(fused_encoding)
            
            geocell = torch.argmax(geocell_logits, dim=1) #get largest of logits
            geocell_coordinate = geocell_to_coordinate(geocell)
            prediction = geocell_coordinate + offset #prediction is geocell + offset.
            
            #haversine loss to evaluate distances from the predictions
            regression_loss = regression_criterion(prediction, target_coordinate)
            haversine_loss += regression_loss.item()
            
            #get biggest proabability geocell from the logits list
            predicted_geocell = torch.argmax(geocell_logits, dim=1)
            true_geocell = coordinates_to_geocell(target_coordinate)
            
            #statistics
            total_correct += (predicted_geocell == true_geocell).sum().item() #if the classification is correct
            distances = haversine_distance(prediction, target_coordinate)  # get haversine distance (not the loss)
            distance_list.extend(distances.cpu().tolist()) #list of all distances
            total_within_100m += (distances <= 0.1).sum().item() #count predictions within 100m
            total_within_1km += (distances <= 1.0).sum().item() #count predictions within 1km
            total_within_5km += (distances <= 5.0).sum().item() #count predictions within 5km
            total_within_10km += (distances <= 10.0).sum().item() #count predictions within 10km
            total_within_50km += (distances <= 50.0).sum().item() #count predictions within 50 km
            total_within_100km += (distances <= 100.0).sum().item() #count predictions within 100 km
            total_within_250km += (distances <= 250.0).sum().item() #count predictions within 250 km

            total_samples += predicted_geocell.shape[0] 
            
    accuracy = total_correct / total_samples
    accuracy100m = total_within_100m / total_samples
    accuracy1km = total_within_1km / total_samples
    accuracy5km = total_within_5km / total_samples
    accuracy10km = total_within_10km / total_samples
    accuracy50km = total_within_50km / total_samples
    accuracy100km = total_within_100km / total_samples
    accuracy250km = total_within_250km / total_samples
    median = np.median(distance_list)

    #print the statistics for the current model weights
    print(f"Validation classification accuracy: {accuracy:.3f} Median: {median:.3f} Within 250km: {accuracy250km:.3f} Within 100km: {accuracy100km:.3f} Within 50km: {accuracy50km:.3f} Within 10km: {accuracy10km:.3f} Within 5km: {accuracy5km:.3f} Within 1km: {accuracy1km:.3f} Within 100m: {accuracy100m:.3f}")
    return haversine_loss / len(validation_dataloader)



''' function predictCoordinates():
Use: 
Predict coordinates of an image with a trained model.

Inputs/outputs:
fused_encoding: the fused encoding generated from the image being predicted
modelName: the model you want to use
returns: The predicted coordinates of the image
'''
def predictCoordinates(fused_encoding, modelName):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    
    #load model weights
    model.load_state_dict(torch.load(f"{modelName}.pth"))
    model.eval()

    with torch.no_grad(): #No gradient computation
        #Use model
        geocell_logits, offset = model(fused_encoding)
    
        geocell = torch.argmax(geocell_logits, dim=1) #get largest of logits
        geocell_coordinate = geocell_to_coordinate(geocell)
        
        #prediction is the offset inside the geocell
        final_prediction = geocell_coordinate + offset
        
    return final_prediction



''' function cleanOCRText():
Use:
Cleans text read from the image OCR, removing any nonsense, punctuation, or camera information. Clean text must be geographic. 

Inputs/outputs:
unclean_list: the image OCR raw output list (text read in an image).
returns: a clean list of words in an image, only consisting of geographic terms such as M45, bridge, London etc.

References:
- Petroules, J. 2010. Regular expression for matching latitude/longitude coordinates? [Online]. [Accessed 8 February 2025]. Available from: https://stackoverflow.com/questions/3518504/regular-expression-for-matching-latitude-longitude-coordinates
- RegExLib. c2001-2025. Regex search for 'latitude'. [Online]. [Accessed 8 February 2025]. Available from: https://regexlib.com/search.aspx?k=latitude+&c=-1&m=-1&ps=20
'''
#Lists and regex for cleaning the OCR text
words_to_delete = [ "NEXTBASE", "NEXTDASE", "YESIMWP", "BLACKVUE", "VIOFO", "GARMIN", "IoIWIPH", "HDR", "HOR", "PRO", "MPH", "ICH", "UHD", "FHD", "NV", "KMVH", "KAE", "AM", "PM"]
coordinate_pattern = r'([NS]?\d{1,3}\.\d{4,})\s*([EW]?\d{1,3}\.\d{4,})?' #very flexible as the data can be misread sometimes, missing either the lat or long section 
no_alpha_pattern = r'^[^a-zA-Z]*$' #will remove any dates or strings with no alphabetical characters
geolocator = Nominatim(user_agent="place_checker")

def cleanOCRText(unclean_list):
    clean_list = []        
    
    for item in unclean_list:
        #remove any trailing or leading symbols
        item = item.strip(" [](){}=@~#_?/<>!*-'\"/")
        
        #skip words in the words_to_delete file (camera information at bottom of screen)
        if any(word.upper() in item.upper() for word in words_to_delete):
            continue
    
        #skip any coordinates, dates, etc. with regex
        if re.search(coordinate_pattern, item) or re.search(no_alpha_pattern, item):
            continue
        
        #remove if contains these symbols
        if "#" in item or "*" in item or "@" in item or "=" in item or '"' in item or "~" in item:
            continue

        #remove if only numbers
        if item.isdigit():
            continue
        
        #remove small words
        if len(item) <= 1:
            continue
        
        #if the ocr item is "A656 Castleford: A642 Garforth", turn it into 4 individual items
        splitted_list = re.split(r'[:;,\-/\s]+', item)
        
        for split_item in splitted_list:
            #This function is slower, so put it last.
            try:
                split_item = split_item.strip(" [](){}=@~#_?/<>!*-'\"/")
                if len(split_item) <=1:
                    continue
                if item.isdigit():
                    continue
                #determine if word is a geographical term, for example "Street" or "London" or "A34".
                location = geolocator.geocode(split_item)
                if location:
                    clean_list.append(split_item)    
            except:
                continue
        
    return clean_list



''' function getTextOCR():
Use:
Using an OCR, reads all text on the screen and cleans it. I am using EasyOCR.

Inputs/outputs:
reader_OCR: the EasyOCR OCR.
img: image to read the text of.
returns: the text read in the image, cleaned.
    
References:
- Jaided AI. 2024. EasyOCR (version 1.7.2). [Software]. [Accessed 8 February 2025]. Available from: https://github.com/JaidedAI/EasyOCR
'''
def getTextOCR(reader_OCR, img):
    textOCRList = reader_OCR.readtext(img, detail = 0) #list of all words read in the image
    textOCRListClean = cleanOCRText(textOCRList)
    textOCR = " ".join(textOCRListClean) #turn list into space seperated string
    return textOCR



''' function preProcessData():
Use:
Preprocess the data for the model's training. Generate the visual encoding, text encoding and fused encoding.

Inputs/outputs:
item: a data item containing an image and caption.
tokenizer_BERT: a bert-base-uncased tokenizer.
model_BERT: a bert-base-uncased model.
processor_ViT: a vit-base-patch16-224-in21k processor.
model_ViT: a vit-base-patch16-224-in21k model.
returns: the fused encoding and target coordinate from the data item.
'''
def preProcessData(item, tokenizer_BERT, model_BERT, processor_ViT, model_ViT):
    img = item.get('image')
    text = item.get('caption')

    #generate embeddings
    text_encoding = getTextEncoding(model_BERT, tokenizer_BERT, text)
    img_encoding = getImageEncoding(model_ViT, processor_ViT, img)
    fused_encoding = crossAttention(img_encoding,text_encoding)
    
    target_coordinate = item.get('computed_geometry').get('coordinates')
    
    return fused_encoding, target_coordinate



''' function blurData():
Use:
Blur all text in an image. Uses an EasyOCR OCR to detect text, and OpenCV to blur.

Inputs/outputs:
img: the image to blur.
reader_OCR: the EasyOCR OCR.
returns: a blurred image.

References:
- Jaided AI. 2024. EasyOCR (version 1.7.2). [Software]. [Accessed 8 February 2025]. Available from: https://github.com/JaidedAI/EasyOCR
- OpenCV. 2025. OpenCV (version 4.11.0). [Software]. [Accessed 1 March 2025]. Available from: https://opencv.org/
- Tammana, S. 2023. Text Detection Using EasyOCR Python. 28 February. Sreekar Tammana's Blog. [Online]. [Accessed 1 March 2025]. Available from: https://sreekartammana.hashnode.dev/text-detection-using-easyocr-python
- deptrai. 2019. Blur content from a rectangle with Opencv. [Online]. [Accessed 1 March 2025]. Available from: https://stackoverflow.com/questions/58163739/blur-content-from-a-rectangle-with-opencv
'''
def blurData(img, reader_OCR):
    try:
        results = reader_OCR.readtext(img)
            
        #get bounding boxes for text read by OCR
        for result in results:
            bbox, text, score = result
            topleft_x_bbox = int(bbox[0][0])
            topleft_y_bbox = int(bbox[0][1])
            bottomright_x_bbox = int(bbox[2][0])
            bottomright_y_bbox = int(bbox[2][1])
            
            #region of interest (y min: y max, x min: x max)
            ROI = img[topleft_y_bbox:bottomright_y_bbox, topleft_x_bbox:bottomright_x_bbox]     
            
            blur = cv2.GaussianBlur(ROI, (51,51), 0)
            
            #(y min: y max, x min: x max)
            img[topleft_y_bbox:bottomright_y_bbox, topleft_x_bbox:bottomright_x_bbox] = blur
    except:
        #the bbox is empty or out of bounds, so simply skip and dont blur anything
        pass
            
    return img



''' function assignCaption():
Use:
Assign captions to the data items - generate a caption with BLIP2, and append OCR text read in the image on the end. 
Also blurs images while the OCR is in use for the blurred model, and removes/cleans any erreneous data items.

Inputs/outputs:
data: the data item (dictionary) with an image inside. 
reader_OCR: the EasyOCR OCR.
model_BLIP2: a blip2-opt-2.7b model
processor_BLIP2: a blip2-opt-2.7b processor
blur: if the image need blurring for the blurred model
singular: Indicate whether one image is being processed through upload, or multiple with Mapillary url links.
returns: 1 on success, meaning the caption in the data dictionary was updated.
'''
def assignCaption(data, reader_OCR, model_BLIP2, processor_BLIP2, blur=False, singular=False):
    invalid_items = []

    #generate captions for images (to avoid doing this manually)
    for i in range(len(data)):
        if singular == False: 
            #working with url links
            try:
                #ensure the data is valid
                img = Image.open(urllib.request.urlopen(data[i].get('thumb_2048_url')))
                img = np.array(img) #convert to np array
                target_coordinate = data[i].get('computed_geometry').get('coordinates')
            except:
                #clean invalid items
                invalid_items.append(i)
                continue
        else: 
            #singular (thumb_2048_url is a filename instead of url)
            img = Image.open(data[i].get('thumb_2048_url')).convert("RGB")
            img = np.array(img) #convert to np array
        
        if blur == True:
            #blur 
            img = blurData(img, reader_OCR)
            print("Image blurred")

        #read any text on screen
        textOCR = getTextOCR(reader_OCR, img)
        #generate caption
        caption = getCaption(model_BLIP2, processor_BLIP2, img)

        #combine caption and OCR text
        if textOCR.strip():  #without whitespace - check if any text exists
            text = caption + " " + textOCR
        else:
            text = caption
        
        data[i]['image'] = img
        data[i]['caption'] = text
        print(f"Generated caption {i+1}/{len(data)}")
        
    #delete any invalid images found (delete in reverse order for index shifts)
    for index in reversed(invalid_items):
        del data[index]
    return 1
        
    
    
''' function main():
Use:
Manages preprocessing, training and predicitng. 
'''
if __name__ == '__main__':
    print(f"[{datetime.now()}] Run started.")
    # Set the random seed for reproducability
    random_seed = 42
    random.seed(random_seed)
    # Set the random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
       
    '''Select which task to perform:'''
    #task = "preprocess"
    #task = "train"
    task = "predict"

    if task == "preprocess":
        #generate data
        data = getData()
        data = data[:350] #limit data to be processed in one go
        
        #data to be used for privacy preserving LVLM
        dataBlurred = copy.deepcopy(data) #deepcopy to ensure all parts of dictionary are copies and not referenced
        print("Data loaded successfully.\n")       
        
        #BLIP2
        #Li, J., Li, D., Savarese, S. and Hoi, S. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML. [Online]. [Accessed 6 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2301.12597
        processor_BLIP2 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model_BLIP2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        model_BLIP2.to(device)
        
        #Jaided AI. 2024. EasyOCR (version 1.7.2). [Online]. [Accessed 8 February 2025]. Available from: https://github.com/JaidedAI/EasyOCR
        reader_OCR = easyocr.Reader(['en'])
        
        #add captions to dictionaries
        assignCaption(data, reader_OCR, model_BLIP2, processor_BLIP2)
        print(f"[{datetime.now()}] data captions assigned.\n")
    
        #blur=True for blurred dataset
        assignCaption(dataBlurred, reader_OCR, model_BLIP2, processor_BLIP2, blur=True)
        print(f"[{datetime.now()}] dataBlurred captions assigned.\n")
    
        #clear memory
        del model_BLIP2 # delete the model object
        torch.cuda.empty_cache() # Clear the GPU memory cache
            
        #Load models for preprocessing
        
        #BERT
        #Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint. [Online]. [Accessed 8 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.1810.04805
        tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
        model_BERT = BertModel.from_pretrained('bert-base-uncased')
        model_BERT.to(device)
        
        #ViT
        #Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. [Online]. [Accessed 12 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2010.11929
        #Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. 2009 IEEE conference on computer vision and pattern recognition. pp.248-255.
        processor_ViT = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        model_ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model_ViT.to(device)
    
        targetCoordinates = []
        fusedEncodings = []
        targetCoordinatesBlurred = []
        fusedEncodingsBlurred = []
        
        #preProcess data:
        
        print("Preprocessing data:")
            
        for i in range(len(data)):
            fused_encoding, target_coordinate = preProcessData(data[i], tokenizer_BERT, model_BERT, processor_ViT, model_ViT)
            fusedEncodings.append(fused_encoding)
            targetCoordinates.append(target_coordinate)
            print(f"Processed {i+1}/{len(data)}")
            
        print("DATA FusedEncodings and targetCoordinates completed successfully.\n")
        print(f"Total lengths are: {len(fusedEncodings)} and {len(targetCoordinates)}\n ")

        #save data:
        
        print("Collecting previous data.\n")
        #save and load lists from pickle file. processing items takes a long time.
        try:
            with open("data.pkl", "rb") as f:
                old_targetCoordinates, old_fusedEncodings = pickle.load(f)
        except:
            old_targetCoordinates, old_fusedEncodings = [], []
                
        targetCoordinates = old_targetCoordinates + targetCoordinates
        fusedEncodings = old_fusedEncodings + fusedEncodings
        print(f"Total lengths are: {len(fusedEncodings)} and {len(targetCoordinates)}\n ")
        
        #save new lists back to file
        with open("data.pkl", "wb") as f:
            pickle.dump((targetCoordinates, fusedEncodings), f)
            
        print(f"[{datetime.now()}] data saved.\n")
        
        #Preprocess blurred data:
            
        print("Preprocessing dataBlurred:")

        for i in range(len(dataBlurred)):
            fused_encoding_blurred, target_coordinate_blurred = preProcessData(dataBlurred[i], tokenizer_BERT, model_BERT, processor_ViT, model_ViT)
            fusedEncodingsBlurred.append(fused_encoding_blurred)
            targetCoordinatesBlurred.append(target_coordinate_blurred)
            print(f"Processed {i+1}/{len(dataBlurred)}")
            
        print("dataBlurred FusedEncodings and targetCoordinates completed successfully.\n")
        print(f"Total lengths are: {len(fusedEncodingsBlurred)} and {len(targetCoordinatesBlurred)}\n ")

        #Save blurred data:

        print("Collecting previous dataBlurred.\n")
        #save and load lists from pickle file. processing items takes a long time.
        try:
            with open("dataBlurred.pkl", "rb") as f:
                old_targetCoordinatesBlurred, old_fusedEncodingsBlurred = pickle.load(f)
        except:
            old_targetCoordinatesBlurred, old_fusedEncodingsBlurred = [], []
                
        targetCoordinatesBlurred = old_targetCoordinatesBlurred + targetCoordinatesBlurred
        fusedEncodingsBlurred = old_fusedEncodingsBlurred + fusedEncodingsBlurred
    
        print(f"Total lengths are: {len(fusedEncodingsBlurred)} and {len(targetCoordinatesBlurred)}\n ")

        #save new lists back to file
        with open("dataBlurred.pkl", "wb") as f:
            pickle.dump((targetCoordinatesBlurred, fusedEncodingsBlurred), f)
    
        print(f"[{datetime.now()}] dataBlurred saved.\n")
    
        #clear memory
        del model_BERT, model_ViT  # Delete the model objects
        torch.cuda.empty_cache()  # Clear the GPU memory cache
        
        
    if task == "train":  
        '''Select baseline or blurred'''
        #model = "baseline"
        model = "blurred"
        
        if model == "baseline":
            print("Collecting data.")
            try:
                with open("data.pkl", "rb") as f:
                    targetCoordinates, fusedEncodings = pickle.load(f)
            except:
                targetCoordinates, fusedEncodings = [], []
            print(f"Total lengths are: {len(fusedEncodings)} and {len(targetCoordinates)}\n ")
    
            #train model
            print("Training baseline model:")
            trainModel(fusedEncodings, targetCoordinates, "baselineLVLM")
            print(f"[{datetime.now()}] baseline model trained successfully.\n")
            
        elif model == "blurred":
            
            print("Collecting dataBlurred.")
            try:
                with open("dataBlurred.pkl", "rb") as f:
                    targetCoordinatesBlurred, fusedEncodingsBlurred = pickle.load(f)
            except:
                targetCoordinatesBlurred, fusedEncodingsBlurred = [], []
            print(f"Total lengths are: {len(fusedEncodingsBlurred)} and {len(targetCoordinatesBlurred)}\n ")
       
            #train model
            print("Training blurred model:")
            trainModel(fusedEncodingsBlurred, targetCoordinatesBlurred, "blurredLVLM")
            print(f"[{datetime.now()}] blurred model trained successfully.\n")
        
    
    if task == "predict":
        '''Select multiple or singular'''
        mode = "multiple"
        #mode = "singular"
        
        #multiple fetches multiple items from Mapillary
        if mode == "multiple":
            #generate new data to predict
            data = getData()
            print("Data loaded successfully.\n")
            data = data[:10]
            
        #singular allows a user to provide an image
        if mode == "singular":
            imagename = "sample.png" #user image
            data = []
            
            #save filename instead
            data.append({
                "thumb_2048_url": imagename,
                #input the coordinates of the image you upload. This is for the distance calculation at the end.
                "computed_geometry": {"coordinates": [0.499283, 51.367190]}
            })
            
        #Load model for captioning
        
        #BLIP2
        #Li, J., Li, D., Savarese, S. and Hoi, S. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML. [Online]. [Accessed 6 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2301.12597
        processor_BLIP2 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model_BLIP2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        model_BLIP2.to(device)
        
        #OCR - easyocr
        #Jaided AI. 2024. EasyOCR (version 1.7.2). [Online]. [Accessed 8 February 2025]. Available from: https://github.com/JaidedAI/EasyOCR
        reader_OCR = easyocr.Reader(['en'])
        
        if mode == "singular":
            #add captions to dictionaries
            assignCaption(data, reader_OCR, model_BLIP2, processor_BLIP2, singular=True)
            print(f"[{datetime.now()}] data captions assigned.\n")
        
        else:
            #add captions to dictionaries
            assignCaption(data, reader_OCR, model_BLIP2, processor_BLIP2)
            print(f"[{datetime.now()}] data captions assigned.\n")

    
        #clear memory
        del model_BLIP2 # Delete the model object
        torch.cuda.empty_cache() #Clear the GPU memory cache
            
        #Load models for preprocessing
        
        #BERT
        #Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint. [Online]. [Accessed 8 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.1810.04805
        tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
        model_BERT = BertModel.from_pretrained('bert-base-uncased')
        model_BERT.to(device)
        
        #ViT
        #Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. [Online]. [Accessed 12 Feburary 2025]. Available from: https://doi.org/10.48550/arXiv.2010.11929
        #Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L. 2009. Imagenet: A large-scale hierarchical image database. 2009 IEEE conference on computer vision and pattern recognition. pp.248-255.
        processor_ViT = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        model_ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model_ViT.to(device)
    
        targetCoordinates = []
        fusedEncodings = []
        
        #preProcess data:
            
        for i in range(len(data)):
            fused_encoding, target_coordinate = preProcessData(data[i], tokenizer_BERT, model_BERT, processor_ViT, model_ViT)
            fusedEncodings.append(fused_encoding)
            targetCoordinates.append(target_coordinate)
            print(f"Processed {i+1}/{len(data)}")
            
        print(f"[{datetime.now()}] data preprocessed.\n")
        
        #clear memory
        del model_BERT, model_ViT # Delete the model objects
        torch.cuda.empty_cache() # Clear the GPU memory cache
        
        target_coordinates = [torch.tensor(target, dtype=torch.float32, device=device) for target in targetCoordinates]
        target_coordinates = torch.stack(target_coordinates) 
        
        #Generate predictions for all models
        for i in range(len(fusedEncodings)):
            print(f"\nData {i+1}:")
            print(f"Image link: {data[i].get('thumb_2048_url')}\n")
            
            setGeocellSize(0.0498) #Due to multiple errors, a smaller value had to be used rather than 0.5. 
            coordinates = predictCoordinates(fusedEncodings[i], "baseline005LVLM")
            distance = haversine_distance(coordinates, target_coordinates[i])
            print(f"""Baseline 0.05 LVLM:: 
                  Predicted: {coordinates.squeeze().tolist()}, 
                  Actual: {target_coordinates[i].squeeze().tolist()}, 
                  Distance: {distance.squeeze().tolist()}""")
        
            setGeocellSize(0.0498) #Due to multiple errors, a smaller value had to be used rather than 0.5.
            coordinates = predictCoordinates(fusedEncodings[i], "blurred005LVLM")
            distance = haversine_distance(coordinates, target_coordinates[i])
            print(f"""Blurred 0.05 LVLM: 
                  Predicted: {coordinates.squeeze().tolist()}, 
                  Actual: {target_coordinates[i].squeeze().tolist()}, 
                  Distance: {distance.squeeze().tolist()}""")
                  
            setGeocellSize(0.098) #Due to multiple errors, a smaller value had to be used rather than 0.1 exactly
            coordinates = predictCoordinates(fusedEncodings[i], "baseline01LVLM")
            distance = haversine_distance(coordinates, target_coordinates[i])
            print(f"""Baseline 0.1 LVLM:
                  Predicted: {coordinates.squeeze().tolist()}, 
                  Actual: {target_coordinates[i].squeeze().tolist()}, 
                  Distance: {distance.squeeze().tolist()}""")
                  
            setGeocellSize(0.098) #Due to multiple errors, a smaller value had to be used rather than 0.1 exactly
            coordinates = predictCoordinates(fusedEncodings[i], "blurred01LVLM")
            distance = haversine_distance(coordinates, target_coordinates[i])
            print(f"""Blurred 0.1 LVLM: 
                  Predicted: {coordinates.squeeze().tolist()}, 
                  Actual: {target_coordinates[i].squeeze().tolist()}, 
                  Distance: {distance.squeeze().tolist()}""")



'''
My results:
    
----------------------baseline005LVLM-------------------------
Validation classification accuracy: 0.367 Median: 39.241 Within 250km: 0.712 Within 100km: 0.577 Within 50km: 0.518 Within 10km: 0.419 Within 5km: 0.388 Within 1km: 0.313 Within 100m: 0.230
Epoch 52: Training loss: 0.23659198696521658, Validation Loss: 187.85727056860924
--------------------------------------------------------------
----------------------blurred005LVLM--------------------------
Validation classification accuracy: 0.356 Median: 41.896 Within 250km: 0.722 Within 100km: 0.583 Within 50km: 0.514 Within 10km: 0.416 Within 5km: 0.381 Within 1km: 0.307 Within 100m: 0.221
Epoch 52: Training loss: 0.24018859022354433, Validation Loss: 185.15634474158287
--------------------------------------------------------------
----------------------baseline01LVLM--------------------------
Validation classification accuracy: 0.389 Median: 35.582 Within 250km: 0.730 Within 100km: 0.588 Within 50km: 0.526 Within 10km: 0.421 Within 5km: 0.379 Within 1km: 0.296 Within 100m: 0.171
Epoch 51: Training loss: 1.9001967487099647, Validation Loss: 180.7532443255186
--------------------------------------------------------------
----------------------blurred01LVLM---------------------------
Validation classification accuracy: 0.381 Median: 37.770 Within 250km: 0.728 Within 100km: 0.589 Within 50km: 0.522 Within 10km: 0.413 Within 5km: 0.369 Within 1km: 0.289 Within 100m: 0.131
Epoch 51: Training loss: 2.0401117218900966, Validation Loss: 184.1919592320919
--------------------------------------------------------------

'''