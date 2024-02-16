import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as im
import os
from PIL import Image

# Intial intialisations :P
test_image_path = "saturn.png"
image = cv2.imread(test_image_path)
Datadirectory = "processed_ds/"
Classes = [subfolder for subfolder in os.listdir(Datadirectory)]

# Contouring to find planets
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blurred, 0, 50)
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 100
filtered_contours = [
    contour for contour in contours if cv2.contourArea(contour) > min_contour_area
]

# Using contours to find coordinates for planets
cood_arr = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cood = [int(x), int(y), int(x + w), int(y + h)]
    cood_arr.append(cood)

# Predictions for each coordinate bound box
pred = []
model_path = "planet_identification_model.h5"
model = tf.keras.models.load_model(model_path)
for i in range(len(cood_arr)):
    crop_img = image[cood_arr[i][1] : cood_arr[i][3], cood_arr[i][0] : cood_arr[i][2]]

    # Setting for prediction and adding to array
    img = Image.fromarray(crop_img)
    img.save('crop.jpeg')
    img = im.load_img('crop.jpeg', target_size=(224, 224))
    img_array = im.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    # img = cv2.resize(crop_img, (224, 224))

    # img = img.reshape(1, 224, 224, 3)
    # img_array = np.asarray(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0

    prediction = model.predict(img_array)
    pred.append(Classes[np.argmax(prediction)])

# Using corresponding coorindate array and prediction array
display = image.copy()
for i in range(len(cood_arr)):
    coor = (cood_arr[i][0], cood_arr[i][1])
    cv2.putText(display, pred[i], coor, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("SpaceyAI", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
