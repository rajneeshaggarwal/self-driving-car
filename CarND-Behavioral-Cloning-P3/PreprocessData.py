import numpy as np;
import pandas as pd;
import os;
import pickle
import cv2;
import tensorflow as tf
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags.
flags.DEFINE_integer('height', 20, "Resize image height")
flags.DEFINE_integer('width', 80, "Reisze image width")
flags.DEFINE_integer('channel', 3, "Image channel")

#Brightness manipulation - 40% to 100%
def brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.4 + np.random.uniform(0,0.6)
    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

# Crop and resize the image to 80*20
def cropImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (FLAGS.width,FLAGS.height), interpolation = cv2.INTER_AREA)
    return image

#Preprocess and randomly select the images to have a generalised training data
def process_line(image, steer):
    img_path = image
    img = mpimg.imread(img_path)
    shape = img.shape

    trans_range = 50
    trans_x = trans_range * np.random.uniform() - trans_range/2
    steer_final = float(steer) + trans_x/trans_range* 0.4

    img = brightness(img)
    img = cropImage(img)

    flip_img = np.random.randint(2)
    if flip_img == 0:
        img = cv2.flip(img,1)
        steer_final = -steer_final
    img = np.array(img)

    return img, steer_final

#Traing data generator function,
#input csv line has 3 images center, left and right, take these images and apply the preprocessing
#Add/Substract a offset from the left/right images to have a recovery scenarios, this will cause the car
#to move like a snake. But safer to have the car on the road than outside.
def train_data(csv_file, BATCH):
    _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
    _y = np.zeros(BATCH, dtype=np.float)
    out_idx = 0
    while 1:
        i = np.random.randint(DATA_SIZE)
        line_content = data[i].strip().split(",")
        steer_main = float(line_content[3])
        #center
        img_c,img_c_s = process_line(line_content[0].strip(), steer_main)
        img_c = img_c.reshape(1,IMAGE_ROW,IMAGE_COL,IMAGE_CH)

        #right
        steer_r = steer_main - 0.25
        img_r,img_r_s = process_line(line_content[2].strip(), steer_r)
        img_r = img_r.reshape(1,IMAGE_ROW,IMAGE_COL,IMAGE_CH)

        #left
        steer_l = steer_main + 0.25
        img_l,img_l_s = process_line(line_content[1].strip(), steer_l)
        img_l = img_l.reshape(1,IMAGE_ROW,IMAGE_COL,IMAGE_CH)

        #Check if we've got valid values
        #CENTER
        if abs(img_c_s) < 0.2:
            if abs(img_c_s) > prevent_bias_center:
                if img_c is not None:
                    _x[out_idx] = img_c
                    _y[out_idx] = img_c_s
                    out_idx += 1
                    if out_idx == BATCH:
                        yield _x, _y

                         # Reset the values back
                        _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                        _y = np.zeros(BATCH, dtype=np.float)
                        out_idx = 0
        else:
            if img_c is not None:
                _x[out_idx] = img_c
                _y[out_idx] = img_c_s
                out_idx += 1
                if out_idx == BATCH:
                    yield _x, _y

                     # Reset the values back
                    _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                    _y = np.zeros(BATCH, dtype=np.float)
                    out_idx = 0

        #RIGHT
        if abs(img_r_s) < 0.2:
            if abs(img_r_s) > prevent_bias_center:
                if img_r is not None:
                    _x[out_idx] = img_r
                    _y[out_idx] = img_r_s
                    out_idx += 1
                    if out_idx == BATCH:
                        yield _x, _y

                         # Reset the values back
                        _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                        _y = np.zeros(BATCH, dtype=np.float)
                        out_idx = 0
        else:
            if img_r is not None:
                _x[out_idx] = img_r
                _y[out_idx] = img_r_s
                out_idx += 1
                if out_idx == BATCH:
                    yield _x, _y

                     # Reset the values back
                    _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                    _y = np.zeros(BATCH, dtype=np.float)
                    out_idx = 0

        #LEFT
        if abs(img_l_s) < 0.2:
            if abs(img_l_s) > prevent_bias_center:
                if img_l is not None:
                    _x[out_idx] = img_l
                    _y[out_idx] = img_l_s
                    out_idx += 1
                    if out_idx == BATCH:
                        yield _x, _y

                         # Reset the values back
                        _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                        _y = np.zeros(BATCH, dtype=np.float)
                        out_idx = 0
        else:
            if img_l is not None:
                _x[out_idx] = img_l
                _y[out_idx] = img_l_s
                out_idx += 1
                if out_idx == BATCH:
                    yield _x, _y

                     # Reset the values back
                    _x = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                    _y = np.zeros(BATCH, dtype=np.float)
                    out_idx = 0

#Validation data generator
#Generates the validation set from the training images.
def val_data(csv_file, BATCH):
    _x_val = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
    _y_val = np.zeros(BATCH, dtype=np.float)
    out_idx = 0
    while 1:
        for i in range(DATA_SIZE):
            line_content = data[i].strip().split(",")
            #center
            img = mpimg.imread(line_content[0].strip())
            steer_val = float(line_content[3].strip())
            img_c = cropImage(img)
            img_c = img_c.reshape(1,IMAGE_ROW,IMAGE_COL,IMAGE_CH)
            if img_c is not None:
                _x_val[out_idx] = img_c
                _y_val[out_idx] = steer_val
                out_idx += 1
            if out_idx == BATCH:
                yield _x_val, _y_val
                # Reset the values back
                _x_val = np.zeros((BATCH, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float)
                _y_val = np.zeros(BATCH, dtype=np.float)
                out_idx = 0


def main(_):

    rows = []
    with open('./data/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            rows.append(row)

    h = FLAGS.height;
    w = FLAGS.width;
    ch = flags.channel;

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
