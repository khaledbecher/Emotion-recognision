import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
from tensorflow.keras import Sequential
from keras.layers import Conv2D , Dense , MaxPooling2D , Flatten , Dropout 

# Obtain datasets path
datasets_path = "D:\datasets"

# Obtain training data (images) path
train_data_path= os.path.join(datasets_path,"emotion","train")

# Obtain test data (images) path
test_data_path= os.path.join(datasets_path,"emotion","test")

# Init Image data generator with rescaling
train_generator = ImageDataGenerator(rescale = 1./255)
test_generator = ImageDataGenerator(rescale = 1./255)

# Load and preprocess train data
train_data =train_generator.flow_from_directory(
    train_data_path,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical")
     

# Load and preprocess test data
test_data =test_generator.flow_from_directory(
    test_data_path,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical")


# Create model architecture
model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3),activation = "relu",input_shape = (48,48,1)))
model.add(Conv2D(64,kernel_size = (3,3),activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(128,kernel_size = (3,3),activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128,kernel_size = (3,3),activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(7,activation = "softmax"))


#Compile the model
model.compile(loss = "categorical_crossentropy",optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001 , decay = 1e-6), metrics = ["accuracy"])


# Train the model
result = model.fit_generator(
    train_data,
    steps_per_epoch = len(train_data),
    epochs = 50,
    validation_data = test_data,
    validation_steps =len(test_data))

# Save model in json 
json_model = model.to_json()
with open("emotion.json","w") as json_file:
    json_file.write(json_model)
    
# Save weghts
model.save_weights("emotion_weights.h5")



