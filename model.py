'''
Created on Sep 4, 2017

@author: asad
'''
import argparse
from sys import exit
import os
import numpy as np
import random
import matplotlib
from csv import reader
from cv2 import imread
from sklearn.utils import shuffle
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Cropping2D, BatchNormalization, Conv2D

def nvidia_model(ch, row, col):
        
    model = Sequential()
    bottom = 20
    top = 50
    model.add(Cropping2D(cropping=((top,bottom),(0,0)),input_shape=(row, col, ch)))
    row = row - top, bottom
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
    
def generator(folder, samples, batch_size=32, training=False):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = folder + '/IMG/'  +batch_sample[0].split('/')[-1]
                center_image = imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if training:
                    #Add flipped centre images
                    image_flipped = np.fliplr(center_image)
                    measurement_flipped = -center_angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)
                    
                    correction = 0.35 # this is a parameter to tune
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                    
                    leftname  = folder + '/IMG/' +batch_sample[1].split('/')[-1]
                    rightname = folder + '/IMG/' +batch_sample[2].split('/')[-1]
                    
                    images.append(imread(leftname))
                    angles.append(steering_left)
                    
                    images.append(imread(rightname))
                    angles.append(steering_right)                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

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

def read_driving_data(folder):

    low_steering_threshold = 0
    drop_threshold = 7
    
    samples = []
    filename = folder + '/driving_log.csv'
    with open(filename) as csvfile:
        cvsreader = reader(csvfile)
        for line in cvsreader:
            steering_angle = abs(float(line[3]))
            if(steering_angle > low_steering_threshold):
                samples.append(line)
            # Randomly drop % of samples with low angles
            elif (random.uniform(0,10) > drop_threshold):
                samples.append(line)
                    
    return samples
   
def main():
    
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to training image folder.'
    )
    args = parser.parse_args()
    
    if args.image_folder != '':
        if not os.path.exists(args.image_folder):
            print("Training folder does not exists.")
            exit(0)
    else:
        print("Training data folder not provided.")
        exit(0)
            
    transfer_learning = True

    samples = read_driving_data(args.image_folder)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    batch_size = 16    
    # compile and train the model using the generator function
    train_generator = generator(args.image_folder, train_samples, batch_size=batch_size, training=True)
    validation_generator = generator(args.image_folder, validation_samples, batch_size=batch_size)
        
    ch, row, col = 3, 160, 320  # Trimmed image format
    
    epochs = 6
    if transfer_learning:
        model = load_model('model.h5')
        sgd = SGD(lr=0.001)
        model.compile(loss='mse', optimizer=sgd)   
        epochs=1     
    else:
        model = nvidia_model(ch, row, col)

        adam = Adam(lr=0.0001)
        model.compile(loss='mse', optimizer=adam)
    
    training_data_length = len(train_samples) * 4 # flipped cente plus left right
    history = model.fit_generator(train_generator, steps_per_epoch= training_data_length/batch_size, 
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples)/batch_size, epochs=epochs,
                        verbose=1)

    
    model.save('model.h5')
    print("Saved model.h5")
    
    plot(history)
    
    return

if __name__ == '__main__':
    main()
