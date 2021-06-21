import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D, BatchNormalization, MaxPooling2D
import matplotlib.pyplot as plt


image_datapath = "./data/IMG/"
correction = 0.2
samples = []

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # read in images and steering angles for the batch
            for batch_sample in batch_samples:
                center_image = process_image(mpimg.imread(image_datapath + batch_sample[0].split('/')[-1]))
                left_image = process_image(mpimg.imread(image_datapath + batch_sample[1].split('/')[-1]))
                right_image = process_image(mpimg.imread(image_datapath + batch_sample[2].split('/')[-1]))
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.extend([center_image,left_image,right_image])
                angles.extend([center_angle,left_angle,right_angle])

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1)) # flipping image for data augmentation
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

def load_data(batch_size=32):
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    return train_generator, validation_generator, train_samples, validation_samples


def KnightRider():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

def main():

    # Hyperparameters
    learning_rate = 0.0001
    batch_size = 128
    epochs = 11

    print('Loading data...')
    train_generator, validation_generator, train_samples, validation_samples = load_data(batch_size)

    print('Building...')
    model = KnightRider()

    print('Compiling...')
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    print('Training...')
    history_object = model.fit_generator(train_generator,
                            steps_per_epoch = np.ceil(len(train_samples)/batch_size),
                            validation_data = validation_generator,
                            validation_steps = np.ceil(len(validation_samples)/batch_size),
                            epochs=epochs,
                            verbose=1)

    print('Saving model...')
    model.save('model.h5')

    print('Model saved, training complete!!!')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig('history.png', dpi=300)

if __name__ == '__main__':
    main()