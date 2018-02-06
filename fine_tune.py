from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
import numpy as np
import cv2
import os
import datetime
import time
from random import shuffle

#LOAD_MODEL = True
MODEL_NAME = 'retrainv2.h5'
#PREV_MODEL = ''

EPOCHS = 1
WIDTH = 480
HEIGHT = 270
FILE_I_START = 149
FILE_I_END = 180
TRAINING_DATA = 'training_data-{}.npy'
CLASS_WEIGHT = {0: 0.21152324,
                1: 3.68784202,
                2: 3.43072156,
                3: 4.50843514,
                4: 2.63336731,
                5: 3.1484867,
                6: 33.76906318,
                7: 20.5026455,
                8: 0.36870525}

def pretty_time_left(time_start, iterations_finished, total_iterations):   
    if iterations_finished == 0:
        time_left = 0
    else:
        time_end = time.time()
        diff_finished = time_end - time_start
        time_per_iteration = diff_finished / iterations_finished
        assert time_per_iteration >= 0
        
        iterations_left = total_iterations - iterations_finished
        assert iterations_left >= 0
        time_left = int(round(iterations_left * time_per_iteration))

    return pretty_dur(time_left)

   
def pretty_running_time(time_start):
    time_end = time.time()
    diff = int(round(time_end - time_start))

    return pretty_dur(diff)

def split_secs(ts_secs):
    dt = datetime.datetime.utcfromtimestamp(ts_secs)
    h, m, s, ms, us = split_datetime(dt)
    return h, m, s, ms, us

def split_datetime(dt):
    h, m, s, us = dt.hour, dt.minute, dt.second, dt.microsecond
    ms = int(round(us / 1000))
    us = us % 1000

    return h, m, s, ms, us


def pretty_dur(dur, fmt_type='full'):
    assert fmt_type in 'minimal, compressed, full'.split(', ')
    
    assert dur >= 0
    h, m, s, ms, us = split_secs(dur)

    if fmt_type == 'minimal':
        dur_str = '{:0>2}:{:0>2}:{:0>2}.{:0>3}'.format(h, m, s, ms)
    elif fmt_type == 'compressed':
        dur_str = '{:0>2}h {:0>2}m {:0>2}.{:0>3}s'.format(h, m, s, ms)
    else:
        dur_str = '{:0>2} hours {:0>2} mins {:0>2} secs {:0>3} msecs'.format(h, m, s, ms)

    return dur_str

logger = keras.callbacks.CSVLogger('./log/'+MODEL_NAME+'.csv', separator=',', append=True)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 9 classes
predictions = Dense(9, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs

time_start = time.time()
training_steps = (FILE_I_END - FILE_I_START) * EPOCHS
j = 0
for e in range(EPOCHS):
    data_order = [i for i in range(FILE_I_START,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        try:
            file_name = TRAINING_DATA.format(i)
            # full file info
            train_data = np.load(file_name)
            print(TRAINING_DATA.format(i),len(train_data))
            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            Y = np.array([i[1] for i in train])

            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = np.array([i[1] for i in test])
            model.fit(x=X, y=Y, batch_size=16, epochs=4, verbose=1, callbacks=[logger], validation_split=0.0, validation_data=(test_x,test_y), 
                shuffle=True, class_weight=CLASS_WEIGHT, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

            j = j+1
            time_passed = pretty_running_time(time_start)
            time_left = pretty_time_left(time_start, j, training_steps)
            print ('Time passed: {}. Time left: {}'.format(time_passed, time_left))
                    
        except Exception as e:
            print(str(e))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
time_start = time.time()
training_steps = (FILE_I_END - FILE_I_START) * EPOCHS
j = 0
for e in range(EPOCHS):
    data_order = [i for i in range(FILE_I_START,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        try:
            file_name = TRAINING_DATA.format(i)
            # full file info
            train_data = np.load(file_name)
            print(TRAINING_DATA.format(i),len(train_data))
            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            Y = np.array([i[1] for i in train])

            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = np.array([i[1] for i in test])
            #model.fit({'input': X}, {'targets': Y}, n_epoch=1, batch_size=16, validation_set=({'input': test_x}, {'targets': test_y}), 
            #     snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
            model.fit(x=X, y=Y, batch_size=16, epochs=4, verbose=1, callbacks=[logger], validation_split=0.0, validation_data=(test_x,test_y), 
                shuffle=True, class_weight=CLASS_WEIGHT, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

            j = j+1
            time_passed = pretty_running_time(time_start)
            time_left = pretty_time_left(time_start, j, training_steps)
            print ('Time passed: {}. Time left: {}'.format(time_passed, time_left))
                    
        except Exception as e:
            print(str(e))