import numpy as np
import os

data_set = []
data_set_np = []
data_set_np_return = []
data_set_T = []
data_set_T_return = []
path_train = "E:/git clone/Federated-Learning-Based-Intrusion-Detection-System/non-iid_data"
path_test = "E:/git clone/Federated-Learning-Based-Intrusion-Detection-System/output_data_T/non-iid_TestData"


def data_load():
    a = os.listdir(path_train)
    for j in a:
        if os.path.splitext(j)[1] == ".csv":
            data_set.append(j)
    for data_name in data_set:
        with open(f"{path_train}/{data_name}", "r+") as f:
            data_set_np = np.loadtxt(f, delimiter=",")
            data_set_np_return.append(data_set_np)
            f.close()

    with open(f"{path_test}.csv","r+") as f_T:
        data_set_T = np.loadtxt(f_T, delimiter=",")
        data_set_T_return.append(data_set_T)
        f_T.close()

    return data_set_np_return,data_set_T_return