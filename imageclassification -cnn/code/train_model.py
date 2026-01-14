# Image Classification using CNN
# Training Script

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
# Dataset folder structure expected:
# dataset/
# ├── train/
# │   ├── cats/
# │   └── dogs/
# └── test/
#     ├── cats/
#     └── dogs/

train_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

# -------------------------------
# Normalize Images
# -------------------------------
def normalize(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(normalize)
validation_ds = validation_ds.map(normalize)

# -------------------------------
# Build CNN Model
# -------------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# -------------------------------
# Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds
)

# -------------------------------
# Plot Accuracy Graph
# -------------------------------
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# -------------------------------
# Save Model
# -------------------------------
model.save('cnn_dog_cat_model.h5')

