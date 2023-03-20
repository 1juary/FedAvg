import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows=0, encoding='utf-8-sig')
    return data
# np.loadtxt(filepath,delimiter,usecols,unpack)
# filepath:加载文件路径
# delimiter:加载文件分隔符

def dataset_pre(data, TL):
    Data_2 = []
    size = np.size(data, axis=0) # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  当没有指定时，返回整个矩阵的元素个数
    for k in range(0, size):
        for j in range(k, k + TL):
            if j < size:
                Data_2.append(data[j])
            else:
                Data_2.append(data[size-1])
    Features = np.array(Data_2)    # 统一一下格式
    return Features