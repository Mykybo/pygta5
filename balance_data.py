# balance_data.py

import numpy as np
import pandas as pd
import os
from collections import Counter
from random import shuffle


starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)
    file_name_bal = 'bal_training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        train_data = np.load(file_name)

        df = pd.DataFrame(train_data)
        #print(df.head())
        print(Counter(df[1].apply(str)))

        lefts = []
        rights = []
        forwards = []

        shuffle(train_data)

        for data in train_data:
            img = data[0]
            choice = data[1]

            if choice == [0,0,1,0,0,0,0,0,0] or choice == [0,0,0,0,1,0,0,0,0] or choice == [0,0,0,0,0,0,1,0,0]:
                lefts.append([img,choice])
            elif choice == [1,0,0,0,0,0,0,0,0]:
                forwards.append([img,choice])
            elif choice == [0,0,0,1,0,0,0,0,0] or choice == [0,0,0,0,0,1,0,0,0]:
                rights.append([img,choice])
            else:
                print('no matches')


        forwards = forwards[:len(lefts)][:len(rights)]
        lefts = lefts[:len(forwards)]
        rights = rights[:len(forwards)]

        final_data = forwards + lefts + rights
        shuffle(final_data)

        np.save(file_name_bal, final_data)
        starting_value += 1
    else:
        print("all done")
        break