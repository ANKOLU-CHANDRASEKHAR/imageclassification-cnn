# Image Classification using CNN
# Prediction Script

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load Trained Model
# -------------------------------
model = tf.keras.models.load_model('cnn_dog_cat_model.h5')

# -------------------------------
# Load and Preprocess Test Image
# -------------------------------
# Replace 'test_image.jpg' with your image name
img_path = 'test_image.jpg'

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))
img = img / 255.0

# Prepare image for prediction
input_img = np.reshape(img, (1, 256, 256, 3))

# -------------------------------
# Predict
# -------------------------------
prediction = model.predict(input_img)

# -------------------------------
# Display Image
# -------------------------------
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# -------------------------------
# Result
# -------------------------------
if prediction[0][0] < 0.5:
    print("Prediction: CAT")
else:
    print("Prediction: DOG")
