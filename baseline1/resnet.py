from keras import backend as K
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Dropout
# from keras.layers import Input
# from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import cv2
import keras
import numpy as np
import os
import pickle
import tensorflow as tf
# Get the path to the images
import argparse
from keras.models import load_model
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train, test, or deploy the model")
parser.add_argument("--mode", type=str,  choices=["train", "test", "deploy"], help="Mode selection", default="train")
args = parser.parse_args()

# images_path = "/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/datasets/fvc2006DB3_A"
# model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/texture_model.h5'

images_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code/datasets/FVC2006DB2_A'
model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code/baseline1/models/resnetcbamnew.h5'

def random_distortion(image):
    rows, cols = image.shape[:2]
    # Define the distortion parameters
    random_scale = np.random.uniform(0.8, 1.2)
    random_angle = np.random.uniform(-15, 15)
    random_translation = np.random.uniform(-10, 10, (2,))
    # Apply the distortion
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, random_scale)
    M[:, 2] += random_translation
    distorted_image = cv2.warpAffine(image, M, (cols, rows))
    # print(distorted_image.shape)
    return distorted_image


def random_gaussian_blurring(image):
    # Generate random blurring parameters
    random_sigma = np.random.uniform(0, 2.0)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (0, 0), random_sigma)
    # print(blurred_image.shape)
    return blurred_image


def random_rotation(image):
    rows, cols = image.shape[:2]
    # Generate random rotation angle
    random_angle = np.random.uniform(-30, 30)
    # Apply rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    # print(rotated_image.shape)
    return rotated_image


def random_scaling(image):
    rows, cols = image.shape[:2]
    # Generate random scaling factor
    random_scale = np.random.uniform(0.8, 1.2)
    # Apply scaling
    scaled_image = cv2.resize(
        image, (int(cols * random_scale), int(rows * random_scale))
    )
    scaled_image = cv2.resize(scaled_image, (cols, rows))
    # print(scaled_image.shape)
    return scaled_image


def random_contrast(image):
    # Generate random contrast factor
    random_factor = np.random.uniform(0.5, 1.5)
    # Apply contrast adjustment
    adjusted_image = np.clip(image * random_factor, 0, 255).astype(np.uint8)
    # print(adjusted_image.shape)
    return adjusted_image


def random_noise(image):
    # Generate random noise parameters
    random_mean = 0
    random_std = np.random.uniform(0, 30)
    # Apply Gaussian noise
    noise = np.random.normal(random_mean, random_std, image.shape).astype(np.uint8)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # print(noisy_image.shape)
    return noisy_image


def random_morphology(image):
    # Generate random kernel size for morphology operations
    kernel_size = np.random.randint(2, 7)
    # Generate random morphology operation
    morph_op = np.random.choice([cv2.MORPH_OPEN, cv2.MORPH_CLOSE])
    # Create the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Apply morphology operation
    morph_image = cv2.morphologyEx(image, morph_op, kernel)
    # print(morph_image.shape)
    return morph_image


# List to store preprocessed images
preprocessed_images_train = []
labels_train = []

preprocessed_images_test = []
labels_test = []

# Iterate over the images
for image_name in os.listdir(images_path):

    if image_name.endswith(".bmp"):
        # Extract the label from the image name without considering variations
        image_name_ex = image_name.split(".")[0]
        # print(image_name_ex)
        image_no = image_name_ex.split("_")[0]
        # print(image_no)
        index = image_name_ex.split("_")[1]
        # print(index)

        # Read the image
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, (224, 224))

        # Convert the image to grayscale
        # gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        normalized_image = resized_image
        # distorted_image = random_distortion(resized_image)
        # blurred_image = random_gaussian_blurring(resized_image)
        # rotated_image = random_rotation(resized_image)
        # scaled_image = random_scaling(resized_image)
        # noisy_image = random_noise(resized_image)
        # adjusted_image = random_contrast(resized_image)
        # morphed_image = random_morphology(resized_image)

        # Save the image
        # processed_image_path = os.path.join(processed_images_path, image_name)
        # cv2.imwrite(processed_image_path, normalized_image)

        # Add the preprocessed image and label to the lists
        if int(index) >= 5 and int(index) <= 12:
            preprocessed_images_train.append(np.array(normalized_image))
            # preprocessed_images_train.append(np.array(distorted_image))
            # preprocessed_images_train.append(np.array(blurred_image))
            # preprocessed_images_train.append(np.array(rotated_image))
            # preprocessed_images_train.append(np.array(scaled_image))
            # preprocessed_images_train.append(np.array(adjusted_image))
            # preprocessed_images_train.append(np.array(noisy_image))
            # preprocessed_images_train.append(np.array(morphed_image))
            labels_train.append(image_no)
            # for _ in range(7):
            #     labels_train.append(image_no)

        else:
            preprocessed_images_test.append(normalized_image)
            labels_test.append(image_no)
            # preprocessed_images_test.extend([distorted_image, blurred_image, rotated_image, scaled_image,adjusted_image,noisy_image,morphed_image])
            # for _ in range(7):
            #     labels_test.append(image_no)
    # break
# Convert the list of preprocessed images to a NumPy array

preprocessed_images_train = np.array(preprocessed_images_train)
preprocessed_images_test = np.array(preprocessed_images_test)


# %%
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
# print(labels_train.shape)
# print(labels_test.shape)

# %%
# Convert labels to numerical labels
label_encoder = LabelEncoder()
numerical_labels_train = label_encoder.fit_transform(labels_train)
numerical_labels_test = label_encoder.fit_transform(labels_test)


# Convert numerical labels to one-hot encoded format
assert numerical_labels_test.shape[0] > 0
assert numerical_labels_train.shape[0] > 0

one_hot_labels_train = to_categorical(numerical_labels_train)
one_hot_labels_test = to_categorical(numerical_labels_test)

# print(one_hot_labels_train)
# print("-========")
# print(one_hot_labels_test)


# %%
margin = 1  # Margin for constrastive loss.
x_train = preprocessed_images_train
y_train = one_hot_labels_train
x_test = preprocessed_images_test
y_test = one_hot_labels_test

# %%
print(f"{x_train.shape=}")
print(f"{y_train.shape=}")
print(f"{x_test.shape=}")
print(f"{y_test.shape=}")

# %%
# print(y_test)

# %%
# https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/



# %%
from keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Dense,
    multiply,
    Permute,
    Concatenate,
    Conv2D,
    Add,
    Activation,
    Lambda,
)



def attach_attention_module(net, attention_module):
    if attention_module == "se_block":  # SE_block
        net = se_block(net)
    elif attention_module == "cbam_block":  # CBAM_block
        net = cbam_block(net)
    else:
        raise Exception(
            "'{}' is not supported attention module!".format(attention_module)
        )

    return net


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(
        channel,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    if K.image_data_format() == "channels_first":
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel, kernel_initializer="he_normal", use_bias=True, bias_initializer="zeros"
    )

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation("sigmoid")(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


# %%
"""
ResNet v1
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""

def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
        x = Dropout(0.5)(x)
    return x

from tensorflow.keras.models import load_model


# # Load the saved model
# saved_model_path = '/home/rs/21CS91R01/research/siamese_network/models/van_minutiae_db2a.h5'
# loaded_model = load_model(saved_model_path)
def resnet_v1(input_shape, depth, num_classes=100, attention_module='se_block',model_path=None):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
   
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            # attention_module
            if attention_module is not None:
                y = attach_attention_module(y, attention_module)
            x = keras.layers.add([x, y])
            x = Activation("relu")(x)
            # x = Dropout(0.5)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    # print(x[0])
    FC = Flatten()(x)  

    feature = Dense(512, activation='relu', name='feature')(FC)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(feature)
    # print("feature shape", feature.shape, "output shape", outputs.shape)
    model = Model(inputs=inputs, outputs=[outputs])

    if model_path:
        model.load_weights(model_path)
    return model


# %%


# print([f.shape for f in features])
# how to get feature vector from last layer of the model
# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer


# %%
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr


# %%


# %%
# Define functions for train, test, and deploy modes
from numpy import expand_dims

def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=50, batch_size=32):
# ... (Same training function as before)

# Train the model and get the history
    depth = 20  # For ResNet, specify the depth (e.g. ResNet50: depth=50)
    model = resnet_v1(input_shape=(224, 224, 3), depth=depth, attention_module='cbam_block',model_path=None, num_classes=140)


    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=["accuracy"],
    )
    model.summary()   
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    return history, model

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

def plot_training_results(history,X_val, y_val, model):
    y_val_pred = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    # Calculate metrics on validation data
    val_precision, val_recall, val_f1 = calculate_metrics(y_true_classes, y_val_pred)

    # Plot train and validation accuracy
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot train and validation loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # # Print and plot precision, recall, and F1-score
    # print(f"Testing Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
    # plt.subplot(2, 1, 2)
    # plt.bar(['Precision', 'Recall', 'F1-score'], [val_precision, val_recall, val_f1])
    # plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

# Assuming you have already prepared your data and defined X_train, y_train, X_val, y_val, and num_classes

# Train the model and get the history
def train():
    history, model = train_model(x_train, y_train, x_test, y_test, num_classes=140, epochs=80, batch_size=32)

    # Plot the training results and metrics
    plot_training_results(history, x_test, y_test, model)

    # model.fit(x_train, y_train,
    #           batch_size=32,
    #           epochs=50,
    #           validation_data=(x_test, y_test),
    #           shuffle=True)
    # # Implement the training code here
    # # Define the path where you want to save the model
    # model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/fvc2006db3a_50epochs.h5'
    model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code/baseline1/models/resnetcbamnew.h5'
    # Save the model using TensorFlow's save method
    model.save_weights(model_path)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])   



#%%
def deploy():
    # Load the saved model
    # saved_model_path = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/fvc2006db3a_50epochs.h5'
    # saved_model = tf.keras.models.load_model(model_path)
    features_dict={}  
   
    # concate_folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/models/resnet_50'

    # folder = '/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/datasets/FVC2006DB2A'
    folder = '/home/mt0/22CS60R42/naf/DB3_A'
    # with open("/home/rs/21CS91R01/research/2023_ICVGIP-Code-1/features_dict_fvc2006_db2a_512.pkl"
    # , "rb") as file:
    #     features_dict = pickle.load(file)
    feature_dict = {}
    for image_name in os.listdir(folder):
    # compare feature_dict keys with image name
        if image_name.endswith('.bmp'):
            # Extract the label from the image name without considering variations
            image_name_ex = image_name.split('.')[0]

            # Check if the image name exists as a key in features_dict
            # if image_name_ex in features_dict:
                # Read the image
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            
            # Resize the image
            resized_image = cv2.resize(image, (224, 224))

                # Get the corresponding features from features_dict
                # features = features_dict[image_name_ex]
                # print("features",features.shape)
                # Predict using your model on the resized image
            model = resnet_v1(input_shape=(224, 224, 3), depth=20, attention_module=None,model_path=model_path)
            
            
            feature_model = Model(inputs=model.input, outputs=model.get_layer('feature').output)
            feature_maps = feature_model.predict(np.array([resized_image]))
                # print(y.shape)
                # Concatenate the features with y
            features_dict[image_name] = feature_maps
            
            # concatenated_features = y
            # # concatenated_features = np.concatenate([features, y], axis=1)
            # print("Concatenated Features:,Name,ShAPE", concatenated_features,image_name_ex,concatenated_features.shape)
            #                 # Save concatenated features to a text file
            # reshaped_features = np.reshape(concatenated_features, (32, 32, 1))
            # output_file_path = os.path.join(concate_folder, f"{image_name_ex}.txt")
            # np.save(output_file_path, reshaped_features)

    with open('features1024.pkl', 'wb') as f:
        pickle.dump(feature_dict, f)

 
            
# %%
def main():
    # Run the selected mode
    if args.mode == "train":
        train()

    elif args.mode == "deploy":
        deploy()
    else:
            pass

if __name__ =='__main__':
    # args
    main()



# %%
