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

TL = 4  # 数字大了直接内存溢出


trainPath_201 = "data/UNSW_NB15_Train201.csv"
trainPath_202 = "data/UNSW_NB15_Train202.csv"
trainPath_203 = "data/UNSW_NB15_Train203.csv"
trainPath_204 = "data/UNSW_NB15_Train204.csv"
trainPath_205 = "data/UNSW_NB15_Train205.csv"

testPath_2 = 'data/UNSW_NB15_TestBin.csv'

trainData_201 = loadCsv(trainPath_201)  # np读取数据
trainData_202 = loadCsv(trainPath_202)
trainData_203 = loadCsv(trainPath_203)
trainData_204 = loadCsv(trainPath_204)
trainData_205 = loadCsv(trainPath_205)

testData_2 = loadCsv(testPath_2)

trainData01_scaler = trainData_201[:, 0:196]  # 筛选行列 arr[r1:r2, c1:c2]
trainData02_scaler = trainData_202[:, 0:196]
trainData03_scaler = trainData_203[:, 0:196]
trainData04_scaler = trainData_204[:, 0:196]
trainData05_scaler = trainData_205[:, 0:196]

testData_scaler = testData_2[:, 0:196]

scaler = MinMaxScaler()

# MinMaxScaler的基本上都应该理解数据归一化，本质上是将数据点映射到了[0,1]区间（默认）
# 但实际使用的的时候也不一定是到[0,1]，你也可以指定参数feature_range

# 仍然属于数据预处理的阶段
scaler.fit(trainData01_scaler)
trainData01_scaler = scaler.transform(trainData01_scaler) # 在Fit的基础上，进行标准化，降维，归一化等操作
scaler.fit(trainData02_scaler)
trainData02_scaler = scaler.transform(trainData02_scaler)
scaler.fit(trainData03_scaler)
trainData03_scaler = scaler.transform(trainData03_scaler)
scaler.fit(trainData04_scaler)
trainData04_scaler = scaler.transform(trainData04_scaler)
scaler.fit(trainData05_scaler)
trainData05_scaler = scaler.transform(trainData05_scaler)

scaler.fit(testData_scaler)
testData_scaler = scaler.transform(testData_scaler)


x_train01 = dataset_pre(trainData01_scaler, TL)
x_train01 = np.reshape(x_train01, (-1, TL, 196)) # -1 是缺省值  满元素除 TL 和 196 就是 -1 的值

x_train02 = dataset_pre(trainData02_scaler, TL)
x_train02 = np.reshape(x_train02, (-1, TL, 196))

x_train03 = dataset_pre(trainData03_scaler, TL)
x_train03 = np.reshape(x_train03, (-1, TL, 196))

x_train04 = dataset_pre(trainData04_scaler, TL)
x_train04 = np.reshape(x_train04, (-1, TL, 196))

x_train05 = dataset_pre(trainData05_scaler, TL)
x_train05 = np.reshape(x_train05, (-1, TL, 196))

x_test = dataset_pre(testData_scaler, TL)
x_test = np.reshape(x_test, (-1, TL, 196))

# Label
y_train01 = trainData_201[:,196]
y_train02 = trainData_202[:,196]
y_train03 = trainData_203[:,196]
y_train04 = trainData_204[:,196]
y_train05 = trainData_205[:,196]

y_test = testData_2[:,196]

shape = np.size(x_train01, axis=2)  # int 值，是有几层数据  196 feature数
# axis的值没有设定，返回矩阵的元素个数
# axis = 0，返回该二维矩阵的行数
# axis = 1，返回该二维矩阵的列数

def nids_model01(shape, serverbs, serverepochs):  # model1 = nids_model01(shape, 500, 1)
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))  # TL = 4 shape = 196
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

    model.summary()  # 打印神经网络结构，统计参数数目

    m = model.get_weights() # 权重矩阵 w 12个 偏置参数b一个 有100列 最后输出一个小数 五个模型都是负数
                            # sequential()有默认初始化的权重和偏置矩阵
    np.save('Server1', m)
    return model

def nids_model02(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
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
                      metrics=['acc'])

    model.fit(x_train02, y_train02, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server2', m)
    return model

def nids_model03(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
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
                      metrics=['acc'])

    model.fit(x_train03, y_train03, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server3', m)
    return model


def nids_model04(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
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
                      metrics=['acc'])

    model.fit(x_train04, y_train04, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server4', m)
    return model

def nids_model05(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
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
                      metrics=['acc'])

    model.fit(x_train05, y_train05, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server5', m)

    return model

def load_models():
    arr = []
    models = glob.glob("*.npy") # 返回所有匹配的文件路径列表
    for i in models:
        arr.append(np.load(i, allow_pickle=True))

    return np.array(arr)

def fl_average():  # FedAvg 方法，聚合参数

    arr = load_models()
    # print(arr)
    fl_avg = np.average(arr, axis=0) # 计算均值，axis = 0 矩阵级别的平均 如不指定，则是全元素平均
    # print(fl_avg)

    return fl_avg

def build_model(avg):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
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
                  metrics=['acc'])

    model.set_weights(avg)   # 设置权重参数 覆盖 sequential()的默认初始化的权重和偏置矩阵
    print("FL Model Ready!")

    return model

def evaluate_model(model, x_test, y_test):
    print('Test Num:', len(y_test))
    score = model.evaluate(x_test, y_test, batch_size=200000, verbose=0)  # 评价函数 例 Score: [0.2843225300312042, 0.9229218363761902]
    # loss,accuracy                             # 这里可以下点功夫，用画图函数表示出来
    print('Score:', score)

def save_fl_model(model):
    model.save("CentralServer/fl_model.h5")

def model_fl():
    avg = fl_average()
    model = build_model(avg)
    evaluate_model(model, x_test, y_test)
    save_fl_model(model)


fl_epochs = 300



for i in range(fl_epochs):

    model1 = nids_model01(shape, 500, 2)
    model2 = nids_model02(shape, 500, 2)
    model3 = nids_model03(shape, 500, 2)
    model4 = nids_model04(shape, 500, 2)
    model5 = nids_model05(shape, 500, 2)
    model_fl()   #  客户模型的Fedavg，并保存
    print('Epoch:', i)

    K.clear_session()