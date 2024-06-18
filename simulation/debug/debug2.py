import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def check_optimal_from_sum_r_list(file: str):
    data = pd.read_csv(file, header=0).to_numpy()
    r = data[:, 1]
    print(np.max(r), np.argmax(r), r.shape)
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/log/pos_stab_train_2_stage_2/'
    check_optimal_from_sum_r_list(path + 'test_record.csv')

    train_r = pd.read_csv(path + 'sumr_list.csv', header=0).to_numpy()
    plt.figure()
    plt.plot(train_r[:, 0], train_r[:, 1])
    plt.show()
