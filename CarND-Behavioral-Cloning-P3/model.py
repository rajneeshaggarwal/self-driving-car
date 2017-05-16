import cv2
import csv
import numpy as np
import sys
from scipy import misc

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, MaxPooling2D, Activation, Reshape
from keras.layers import Convolution2D, Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

images = []
measurements = []
h,w,c=160,320,3

# randomily change the image brightness
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness - referenced Vivek Yadav post
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.resize(image, (w,h))

def readImageData(source_path, steering_angle):
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    image = randomise_image_brightness(image)
    position = filename.split('_')[0]
    if(position == 'left'):
        steering_angle += 0.25
    else:
        steering_angle -= 0.25
    image_flipped = np.fliplr(image)
    measurement_flipped = -steering_angle

    images.append(image)
    images.append(image_flipped)
    measurements.append(steering_angle)
    measurements.append(measurement_flipped)

def get_model():

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(h,w,c)))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    
    model.add(Dropout(.2))
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(BatchNormalization(mode=2))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def main():
    rows = []
    with open('./data/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            rows.append(row)

    for row in rows[2:]:
        steering_angle = float(row[3])
        readImageData(row[0], steering_angle)
        readImageData(row[1], steering_angle)
        readImageData(row[2], steering_angle)

    print("Images Size: ", len(images))
    X_train = np.array(images)
    y_train = np.array(measurements)

    model = get_model()

    model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=int(sys.argv[1]))
    model.save('model.h5')

if __name__ == '__main__':
    main()
