import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from keras import regularizers

print(len(os.listdir("Chest_COVID")))
print(len(os.listdir("Chest_NonCOVID")))

try:
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/training')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/test')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/training/Covid')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/training/NonCovid')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/test/Covid')
    os.mkdir('C:/Users\91797/Desktop/Covid_19_Detector/chest/test/NonCovid')
except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []

    for unitData in os.listdir(SOURCE):
        data = SOURCE + '/' + unitData
        if (os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file i.e zero size')

    train_set_length = int(len(dataset) * SPLIT_SIZE)
    test_set_length = int(len(dataset) - train_set_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:train_set_length]
    test_set = shuffled_set[-test_set_length:]

    for unitData in train_set:
        temp_train_set = SOURCE + "/" + unitData
        final_train_set = TRAINING + "/" + unitData
        copyfile(temp_train_set, final_train_set)

    for unitData in test_set:
        temp_test_set = SOURCE + '/' + unitData
        final_test_set = TESTING + '/' + unitData
        copyfile(temp_test_set, final_test_set)

covid_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/Chest_COVID'
training_covid_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/chest/training/Covid'
testing_covid_dir =  'C:/Users/91797/Desktop/Covid_19_Detector/chest/test/Covid'
non_covid_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/Chest_NonCOVID'
training_non_covid_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/chest/training/NonCovid'
testing_non_covid_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/chest/test/NonCovid'

split_size = .8
split_data(covid_dir,training_covid_dir,testing_covid_dir,split_size)
split_data(non_covid_dir,training_non_covid_dir,testing_non_covid_dir,split_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Training_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/chest/training'
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(Training_dir,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(300, 300))

Validation_dir = 'C:/Users/91797/Desktop/Covid_19_Detector/chest/test'
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(Validation_dir, batch_size=10, class_mode='binary',
                                                              target_size=(300,300))

history = model.fit_generator(train_generator,
                              epochs=40,
                              verbose=1,
                              validation_data=validation_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

