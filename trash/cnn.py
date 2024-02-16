import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


Datadirectory = "processed_ds/"
Classes = [subfolder for subfolder in os.listdir(Datadirectory)]
img_size = 224

training_Data = []  
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_Data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_Data()
# random.shuffle(training_Data)

X = [] 
y = [] 
for features, label in training_Data:
    X.append(features)  
    y.append(label)  

X = np.array(X).reshape(-1, img_size, img_size, 3)  
# Normalizing the data
X = X.astype('float32') / 255.0

Y = np.array(y) 

## BEGIN PRE-PROCESSING##
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)
data = datagen.flow(X, Y, batch_size=16)



# Check if the model weights file exists
model_weights_file = 'Final_model_try.h5'
if os.path.exists(model_weights_file):
    # Load the pre-trained model with saved weights
    new_model = tf.keras.models.load_model(model_weights_file)
    print("Model loaded from file.")
else:
    # Create and train the model
    model = tf.keras.applications.MobileNetV2()
    base_input = model.layers[0].input
    base_output = model.layers[-2].output 
    final_output = base_output
    # final_output = layers.Dense(128)(base_output)     
    # final_output = layers.Activation('relu')(final_output)
    # final_output = layers.Dense(64)(final_output) 
    # final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(len(Classes),activation='softmax',)(final_output)  

    # Add dropout after the dense layer
    # final_output = layers.Dropout(0.5)(final_output)  # Adjust the dropout rate

    new_model = keras.Model(inputs=base_input, outputs=final_output)

    new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    # Training the model
    new_model.fit(data, epochs=2)

    # Save the trained model weights
    new_model.save(model_weights_file)
    # print("Model trained and saved to file.")


# Example: Load and predict on a test image
test_image = cv2.imread("saturn.png")
test_image = cv2.resize(test_image, (img_size, img_size))
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

predictions = new_model.predict(test_image)
print("Predictions:", Classes[np.argmax(predictions)])
print(len(predictions[0]))


