import cv2
import numpy as np

arr = np.load('training_data-1-mini.npy')

for i in range(0,len(arr)):
	print(arr[i][1])
	cv2.imshow('img',arr[i][0])
	if cv2.waitKey(20) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
