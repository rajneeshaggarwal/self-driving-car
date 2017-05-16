import cv2
import csv
import numpy as np
import sys
from scipy import misc
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, MaxPooling2D, Activation, Reshape
from keras.layers import Convolution2D, Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, load_img

images = []
measurements = []
h,w,c=160,320,3

new_size_col,new_size_row = 64, 64

# randomily change the image brightness
def randomise_image_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    return image

def readImageData(source_path, steering_angle):
    filename = source_path.split('/')[-1]

    position = filename.split('_')[0]
    if(position == 'left'):
        steering_angle += 0.25
    else:
        steering_angle -= 0.25

    current_path = './data/IMG/' + filename
    # image = cv2.imread(current_path)
    image = load_img(current_path)
    image = img_to_array(image)

    image = randomise_image_brightness(image)
    image = preprocessImage(image)
    image = np.array(image)
    images.append(image)
    measurements.append(steering_angle)

    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
        images.append(image)
        measurements.append(steering_angle)

def get_model():

    model = Sequential()
    model.add(Lambda(lambda x: x/225.0 - 0.5, input_shape=(new_size_col,new_size_row,c)))

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

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=int(sys.argv[1]))
    model.save('model.h5')

if __name__ == '__main__':
    main()
