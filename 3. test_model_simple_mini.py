import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pyvjoy

j = pyvjoy.VJoyDevice(1)
j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_RX, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_RY, 0x4000)
j.set_axis(pyvjoy.HID_USAGE_RZ, 0x4000)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))

model_type = 'categorical'
MODEL_NAME = 'miniv3.h5'
GAME_WIDTH = 1366
GAME_HEIGHT = 768

WIDTH = 50
HEIGHT = 50

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    # ReleaseKey(W)
    j.set_axis(pyvjoy.HID_USAGE_X, 0x0)
    if random.randrange(0,2) == 0:
        PressKey(W)
def right():
    # ReleaseKey(W)
    j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
    if random.randrange(0,2) == 0:
        PressKey(W)
def forward_left():
    j.set_axis(pyvjoy.HID_USAGE_X, 0x0)
    PressKey(W)
def forward_right():
    j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
    PressKey(W)
def straight():
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    PressKey(W)
    
# def straight():
#     PressKey(W)
#     ReleaseKey(A)
#     ReleaseKey(D)
#     ReleaseKey(S)

# def left():
#     if random.randrange(0,5) == 1:
#         PressKey(W)
#     else:
#         PressKey(A)
#         # ReleaseKey(W)
#         ReleaseKey(S)
#         ReleaseKey(D)

# def right():
#     if random.randrange(0,5) == 1:
#         PressKey(W)
#     else:
#         PressKey(D)
#         # ReleaseKey(W)
#         ReleaseKey(A)
#         ReleaseKey(S)
    
# def reverse():
#     PressKey(S)
#     ReleaseKey(A)
#     ReleaseKey(W)
#     ReleaseKey(D)


# def forward_left():
#     # if random.randrange(0,20) == 1:
#     PressKey(W)
#     PressKey(A)
#     ReleaseKey(D)
#     ReleaseKey(S)
    
    
# def forward_right():
#     PressKey(W)
#     PressKey(D)
#     ReleaseKey(A)
#     ReleaseKey(S)

    
# def reverse_left():
#     PressKey(S)
#     PressKey(A)
#     ReleaseKey(W)
#     ReleaseKey(D)

    
# def reverse_right():
#     PressKey(S)
#     PressKey(D)
#     ReleaseKey(W)
#     ReleaseKey(A)

# def no_keys():

#     if random.randrange(0,3) == 1:
#         PressKey(W)
#     else:
#         ReleaseKey(W)
#     ReleaseKey(A)
#     ReleaseKey(S)
#     ReleaseKey(D)
    
model = load_model(MODEL_NAME)
print('We have loaded a previous model!!!!')

predictions = deque()
for i in range(1):
    predictions.append('2')

def main():
    angle = 0x4000
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0,0,GAME_WIDTH,GAME_HEIGHT))
    screen = screen[664:714,104:154]
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (WIDTH,HEIGHT))
    # cv2.imshow('img',screen)  # check if capture is correct
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,0,GAME_WIDTH,GAME_HEIGHT))
            screen = screen[664:714,104:154]

            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            lower_blue = np.array([134,0,0])
            upper_blue = np.array([137,255,255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            screen = cv2.bitwise_and(screen,screen, mask= mask)

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            last_time = time.time()
            screen = screen.reshape((1,) + screen.shape)
            
            prediction = model.predict(screen, batch_size=None, verbose=0, steps=None)
            print(prediction)
            if model_type == 'binary':
                if prediction[0][0] < 0.5:
                    mode_choice = 0
                else:
                    mode_choice = 1

                if mode_choice == 0:
                    left()
                    
                elif mode_choice == 1:
                    right()

                else:
                    print("Unknown command")
            else:
                prediction = np.argmax(prediction)
                predictions.append(prediction)
                predictions.popleft()
                lefts = predictions.count(0) + predictions.count(3)
                rights = predictions.count(1) + predictions.count(4)
                # print(predictions)
                print(prediction)
                if prediction == 0 or prediction == 3:
                    angle = 0x0
                elif prediction == 1 or prediction == 4:
                    angle = 0x8000
                else:
                    angle = 0x4000
                j.set_axis(pyvjoy.HID_USAGE_X, angle)
                j.set_axis(pyvjoy.HID_USAGE_RZ, 0x5000)
                time.sleep(0.01)
                j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
                # if prediction == 0:
                #     left()
                # elif prediction == 1:
                #     right()
                # elif prediction == 2:
                #     straight()
                # elif prediction == 3:
                #     forward_left()
                # elif prediction == 4:
                #     forward_right()
                # else:
                #     print("wat")

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                j.set_axis(pyvjoy.HID_USAGE_RZ, 0x4000) # Throttle
                j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)  # Steering
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
