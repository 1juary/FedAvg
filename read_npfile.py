import numpy as np

loadData1 = np.load('Server1.npy', allow_pickle=True)
loadData2 = np.load('Server2.npy', allow_pickle=True)
loadData3 = np.load('Server3.npy', allow_pickle=True)
loadData4 = np.load('Server4.npy', allow_pickle=True)
loadData5 = np.load('Server5.npy', allow_pickle=True)

print(loadData1.shape)
print(loadData2.shape)
print(loadData3.shape)
print(loadData4.shape)
print(loadData5.shape)

# print(loadData1)
# print(loadData2)
# print(loadData3)
# print(loadData4)
print(loadData5)
