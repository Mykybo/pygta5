import numpy as np
import cv2
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

# model = googlenet(WIDTH, HEIGHT, 3, LR, output=9)
# MODEL_NAME = 'model_smallv3.model'
# model.load(MODEL_NAME)
model = load_model('retrainv2.h5')

print('We have loaded a previous model!!!!')

arr = np.load('training_data-1.npy')
for i in range(0,len(arr)):
	if i == 500:
		cv2.destroyAllWindows()
		break
	
	cv2.imshow('img',arr[i][0])
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
											  
	screen = arr[i][0]
	screen = cv2.resize(screen,(WIDTH,HEIGHT))
	#screen = np.ones(shape=(WIDTH,HEIGHT,3))
	screen = np.reshape(screen,[1,WIDTH,HEIGHT,3])
	prediction = model.predict(screen, batch_size=None, verbose=0, steps=None)
	# prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
	# prediction = np.array(prediction) * np.array([1,1,1,1,1,1,1,1,1])
	# prediction = np.array(prediction)
	
	# print(i,prediction,np.argmax(prediction))
	# print(i,arr[i][1])
	mode_choice = np.argmax(prediction)
	if mode_choice == 0:
		choice_picked = 'straight'
	elif mode_choice == 1:
		choice_picked = 'reverse'
	elif mode_choice == 2:
		choice_picked = 'left'
	elif mode_choice == 3:
		choice_picked = 'right'
	elif mode_choice == 4:
		choice_picked = 'forward+left'
	elif mode_choice == 5:
		choice_picked = 'forward+right'
	elif mode_choice == 6:
		choice_picked = 'reverse+left'
	elif mode_choice == 7:
		choice_picked = 'reverse+right'
	elif mode_choice == 8:
		choice_picked = 'nokeys'
	print(choice_picked)
