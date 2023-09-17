import os
import cv2
import numpy as np
import pickle

import tensorflow as tf
import numpy as np

# Load the data using NumPy (Assuming 'features.pt' contains a dictionary)
# dic = np.load('/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/van_features_fvc2006DB2A.pt', allow_pickle=True).item()

import torch

dic = torch.load('/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/van_features_fvc2006DB2A.pt')


for key, value in dic.items():
    print(key, len(value))


# # serialized_dict = "\n".join([f"{key}: {value}" for key, value in features_dict.items()])
# import pickle
# with open("van_features.pkl", "wb") as file:
#     pickle.dump(dic, file)



concate_folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/van_minutiae_fvc2006db3a'
# Load the dictionary from the pickle file
with open("/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/512_minutiae_features_dict_fvc2006_db3a_.pkl", "rb") as file:
    min_features_dict = pickle.load(file)

with open("/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/van_features.pkl", "rb") as file:
    van_features_dict = pickle.load(file)
# folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/datasets/fvc2006DB3_A'

import numpy as np
from keras.models import load_model

# # Step 1: Load the saved model
# saved_model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/texture_model.h5'
# loaded_model = load_model(saved_model_path)
for key, value in van_features_dict.items():
    if key.endswith('.bmp'):
        key_ex = key.split('.')[0]
        if key_ex in min_features_dict:
            min_features = min_features_dict[key_ex]
            van_features = np.array(value)
            van_features = np.expand_dims(van_features, axis=0)  
            print("min_features",min_features.shape)

            print("van_features",(van_features.shape))
            # print(min_features)
            # print(van_features)
            concatenated_features = np.concatenate([min_features, van_features], axis=1)
            print(",Name,ShAPE",key_ex,concatenated_features.shape)
            
            # concatenated_features = np.concatenate([min_features, van_features], axis=1)
            # print("Concatenated Features:,Name,ShAPE", concatenated_features,key_ex,concatenated_features.shape)
            #             # Save concatenated features to a text file
            reshaped_features = np.reshape(concatenated_features, (32,32, 1))
            output_file_path = os.path.join(concate_folder, f"{key_ex}.txt")
            np.save(output_file_path, reshaped_features)
            

# # for image_name in os.listdir(folder):
# #     # compare feature_dict keys with image name
# #     if image_name.endswith('.bmp'):
# #         # Extract the label from the image name without considering variations
# #         image_name_ex = image_name.split('.')[0]

# #         # Check if the image name exists as a key in features_dict
# #         if image_name_ex in features_dict:
# #             # # Read the image
# #             # image_path = os.path.join(folder, image_name)
# #             # image = cv2.imread(image_path)
        
# #             # # Resize the image
# #             # resized_image = cv2.resize(image, (224, 224))

# #             # Get the corresponding features from features_dict
# #             features = features_dict[image_name_ex]

# #             # Predict using your model on the resized image
# #             # [feature_maps, y] = model.predict(np.array([resized_image]))

# #             # Concatenate the features with y
# #             # concatenated_features = np.concatenate([features, y], axis=1)
# #             concatenated_features = features
# #             print("Concatenated Features:,Name,ShAPE", concatenated_features,image_name_ex,concatenated_features.shape)
# #                         # Save concatenated features to a text file
# #             reshaped_features = np.reshape(concatenated_features, (32,32, 1))
# #             output_file_path = os.path.join(concate_folder, f"{image_name_ex}.txt")
# #             np.save(output_file_path, reshaped_features)


import os

# Define the folder path
folder_path = '/content/drive/MyDrive/Minutiae_points_fvc2006db2_fingernet'


# Initialize an empty list to store all data from files
fingerNet_min = []

# List all files in the folder
file_list = os.listdir(folder_path)

# Read each file and extract the data
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = lines[0].split()  # Split the first line to get the labels
        data_list = []
        for line in lines[2:]:
            values = line.split()  # Split each line to get x, y, and orientation values
            x = int(values[0])  # Convert x to an integer (if needed)
            y = int(values[1])  # Convert y to an integer (if needed)
            # orientation = float(values[2])  # Convert orientation to a float (if needed)
            data_list.append([x, y])  # Append the values as a list to the data_list
        fingerNet_min.append((labels, data_list))  # Append the data for each file as a tuple to all_data


import os

# Define the folder path
folder_path = '/content/drive/MyDrive/output_minutiae1'

# Initialize an empty list to store all data from files
traditional_minu = []

# List all files in the folder
file_list = os.listdir(folder_path)

# Read each file and extract the data
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = lines[0].split()  # Split the first line to get the labels
        data_list = []
        for line in lines[1:]:
            values = line.split()  # Split each line to get x, y, and orientation values
            x = int(values[0])  # Convert x to an integer (if needed)
            y = int(values[1])  # Convert y to an integer (if needed)
            # orientation = float(values[2])  # Convert orientation to a float (if needed)
            data_list.append([x, y])  # Append the values as a list to the data_list
        traditional_minu.append((labels, data_list))  # Append the data for each file as a tuple to all_data



for i, (trad_labels, trad_data_list) in enumerate(traditional_minu, 1):
  for i, (minu_labels, minu_data_list) in enumerate(fingerNet_min, 1):
    if trad_labels[0] == minu_labels[0]:
      print(trad_labels[0])
      print(trad_data_list)
      print(minu_data_list)
      print("----------------------------------------------------------")