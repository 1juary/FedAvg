import glob
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tensorflow import keras
import warnings
from dataPre import loadCsv, dataset_pre
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=Warning) # 取消警告
from data_load import data_load
from model_train import model_train

TL = 1
shape = 115
serverbs, serverepochs = 300, 1

if __name__ == "__main__":

    data_set, test_set = data_load()
    print("data_set prepared")
    data_set_feature = []
    data_set_label = []
    model = []


    # 分割feature and label  由于数据已经预处理过了，这里不再处理
    for cnt in range(len(data_set)) :
        data_feature = data_set[cnt][:, 0:115]
        data_feature = np.reshape(data_feature, (-1, TL, 115))
        data_label = data_set[cnt][:, 115]
        # data_label = np.reshape(data_label, (-1, TL, 1))
        # data_set_feature.append(data_feature)
        # data_set_label.append(data_label)
        test_feature = test_set[0][:3999, 0:115]
        test_feature = np.reshape(test_feature, (-1, TL, 115))
        test_label = test_set[0][:3999, 115]
        # test_label = np.reshape(test_label, (-1, TL, 1))



        # keras 建模型还是靠类更加合理，函数会出现call()函数参数问题，改天在chatgpt问一下  3.20 今天就到这里吧

        # model_train = model_train(shape, serverbs, serverepochs, TL  #
        #                           , data_feature, data_label, cnt, test_feature, test_label)
        model.append(model_train)



