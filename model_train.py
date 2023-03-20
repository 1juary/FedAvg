import glob
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tensorflow import keras
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=Warning)  # 取消警告


def model_train(shape, serverbs, serverepochs, TL, x_train01, y_train01, num, x_test, y_test):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:

        K.clear_session()

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))  # 1 115
        # 该层创建了一个卷积核，该卷积核以 单个空间（或时间）维上的层输入进行卷积
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc']) # 评估当前训练模型的性能

    model.fit(x_train01, y_train01, batch_size=serverbs, epochs=serverepochs,  #  batch_size	每一个batch的大小（批尺寸），即训练一次网络所用的样本数
                            validation_data=(x_test, y_test), verbose=2, shuffle=True) # epochs	迭代次数，即全部样本数据将被“轮”多少次，轮完训练停止
                            # validation_data 指定验证数据    verbose 日志 =2为每一个epoch输出一条记录   shuffle=True, #布尔值。是否在每轮迭代之前混洗数据

    # model.summary()  # 打印神经网络结构，统计参数数目

    m = model.get_weights() # 权重矩阵 w 12个 偏置参数b一个 有100列 最后输出一个小数 五个模型都是负数
                            # sequential()有默认初始化的权重和偏置矩阵
    np.save(f"./model/Server{num}", m)
    print("model_train prepared")
    return model
