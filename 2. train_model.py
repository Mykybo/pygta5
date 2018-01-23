import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle
import datetime



FILE_I_END = 2

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 2

MODEL_NAME = 'modelv2.model'
PREV_MODEL = 'modelv1.model'

LOAD_MODEL = True

wl = 0
sl = 0
al = 0
dl = 0

wal = 0
wdl = 0
sal = 0
sdl = 0
nkl = 0

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]


model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
    

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

# iterates through the training files

time_start = time.time()
training_steps = FILE_I_END * EPOCHS
j = 0

for e in range(EPOCHS):
    #data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        try:
            file_name = 'training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)
            print('training_data-{}.npy'.format(i),len(train_data))

##            # [   [    [FRAMES], CHOICE   ]    ] 
##            train_data = []
##            current_frames = deque(maxlen=HM_FRAMES)
##            
##            for ds in data:
##                screen, choice = ds
##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
##
##
##                current_frames.append(gray_screen)
##                if len(current_frames) == HM_FRAMES:
##                    train_data.append([list(current_frames),choice])


            # #
            # always validating unique data: 
            #shuffle(train_data)
            train = train_data[:-50]
            test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = [i[1] for i in test]
            model.fit({'input': X}, {'targets': Y}, n_epoch=1, batch_size=16, validation_set=({'input': test_x}, {'targets': test_y}), 
                 snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
           # model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
             #   snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
             

            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

            j = j+1
            time_passed = pretty_running_time(time_start)
            time_left = pretty_time_left(time_start, j, training_steps)
            print ('Time passed: {}. Time left: {}'.format(time_passed, time_left))
                    
        except Exception as e:
            print(str(e))
            
    








#

#tensorboard --logdir=foo:J:/phase10-code/log

