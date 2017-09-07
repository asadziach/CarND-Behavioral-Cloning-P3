'''
Created on Sep 4, 2017

@author: asad
'''
import cv2
import numpy as np
import sklearn
import csv
import random
import matplotlib
from sklearn.utils import shuffle
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Cropping2D, BatchNormalization, Conv2D

def nvidia_model(ch, row, col):
        
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
    row = 90
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(BatchNormalization(epsilon=0.001, axis=1,input_shape=(row, col, ch)))
    
    model.add(Conv2D(24,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(64,(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(Conv2D(64,(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    
    return model    
    
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

def read_csv(filename):

    low_steering_threshold = 0.05
    
    samples = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            steering_angle = abs(float(line[3]))
            if(steering_angle > low_steering_threshold):
                samples.append(line)
            # Randomly drop 70% of samples with low angles
            elif (random.uniform(0,10) > 7):
                samples.append(line)
                    
    return samples
   
def main():
    
    samples = read_csv('data/driving_log.csv')

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    batch_size = 32    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
        
    ch, row, col = 3, 160, 320  # Trimmed image format
    
    model = nvidia_model(ch, row, col)

    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)
    
    #model = load_model('model.h5')
    
    history = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size, 
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples)/batch_size, epochs=30,
                        verbose=1)

    
    model.save('model.h5')
    print("Saved model.h5")
    
    plot(history)
    
    return

if __name__ == '__main__':
    main()