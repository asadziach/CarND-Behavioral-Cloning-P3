'''
Created on Sep 4, 2017

@author: asad
'''

import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/' +batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def plot(history_object):
    import matplotlib
    matplotlib.use('GTK3Cairo',warn=False, force=True)
    import matplotlib.pyplot as plt
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()    
    
def main():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    batch_size = 32    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Convolution2D
    
    ch, row, col = 3, 160, 320  # Trimmed image format
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
    row = 90
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(BatchNormalization(epsilon=0.001, axis=1,input_shape=(row, col, ch)))   #ch, row,col
    
    model.add(Convolution2D(24,5,5, subsample=(2,2) ,border_mode='valid', activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2) ,border_mode='valid', activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2) ,border_mode='valid', activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1) ,border_mode='valid', activation='relu'))
    model.add(Convolution2D(64,3,3, subsample=(1,1) ,border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)
    
    history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=3,
                        verbose=1)

    
    model.save('model.h5')
    print("Saved model.h5")
    
    plot(history)

    # Load data
    # Preprocess
    # Standardize
    # shuffl/split
    
    return

if __name__ == '__main__':
    main()