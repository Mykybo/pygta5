import numpy as np
from sklearn.utils import class_weight

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

arr = np.load('training_data-149.npy'.format(1))
for i in range(150,180):
	print("Loading file {}".format(i))
	arr = np.append(arr,np.load('training_data-{}.npy'.format(i)),axis=0)
outputs = arr[:,1]
y_train = []
for row in outputs:
	if row == w:
		y_train.append(0)
	elif row == s:
		y_train.append(1)
	elif row == a:
		y_train.append(2)
	elif row == d:
		y_train.append(3)
	elif row == wa:
		y_train.append(4)
	elif row == wd:
		y_train.append(5)
	elif row == sa:
		y_train.append(6)
	elif row == sd:
		y_train.append(7)
	elif row == nk:
		y_train.append(8)

for row in y_train:
	print(row)
print(np.unique(y_train))
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print(class_weight)