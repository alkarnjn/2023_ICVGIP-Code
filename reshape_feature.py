import os
import cv2
import numpy as np
import pickle

concate_folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/datasets/combined_fvc2006db2a'
# Load the dictionary from the pickle file
with open("/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/features_dict.pkl", "rb") as file:
    features_dict = pickle.load(file)

folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/datasets/FVC2002DB2A'

import numpy as np
from keras.models import load_model

# Step 1: Load the saved model
saved_model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/texture_model.h5'
loaded_model = load_model(saved_model_path)


for image_name in os.listdir(folder):
    # compare feature_dict keys with image name
    if image_name.endswith('.tif'):
        # Extract the label from the image name without considering variations
        image_name_ex = image_name.split('.')[0]

        # Check if the image name exists as a key in features_dict
        if image_name_ex in features_dict:
            # # Read the image
            # image_path = os.path.join(folder, image_name)
            # image = cv2.imread(image_path)
        
            # # Resize the image
            # resized_image = cv2.resize(image, (224, 224))

            # Get the corresponding features from features_dict
            features = features_dict[image_name_ex]

            # Predict using your model on the resized image
            # [feature_maps, y] = model.predict(np.array([resized_image]))

            # Concatenate the features with y
            # concatenated_features = np.concatenate([features, y], axis=1)
            concatenated_features = features
            print("Concatenated Features:,Name,ShAPE", concatenated_features,image_name_ex,concatenated_features.shape)
                        # Save concatenated features to a text file
            reshaped_features = np.reshape(concatenated_features, (32,32, 1))
            output_file_path = os.path.join(concate_folder, f"{image_name_ex}.txt")
            np.save(output_file_path, reshaped_features)