#!/usr/bin/env python
# coding: utf-8

# # 尝试单个矩阵的SVD分解

# In[4]:


import numpy as np


if __name__ == '__main__':
    M = np.array(
        [
            [0, 4.5, 2.0, 0],
            [4.0, 0, 3.5, 0],
            [0, 5.0, 0, 2.0],
            [0, 3.5, 4.0, 1.0]
        ]
    )
    U, S, V_T = np.linalg.svd(M)
    k = 2 
    # 取前2个奇异值对应的列向量
    # 分别打印
    Vec_user, Vec_item = U[:,:k], V_T[:k, :].T
    print(Vec_user, "\n\n", Vec_item,"\n\n", S)
    print("\n")
    print(U, "\n\n", V_T,"\n\n")


# # 尝试低秩近似

# In[14]:


import numpy as np
from sklearn.decomposition import PCA 
#decomposition随机截断的SVD,取决于输入数据的形状和要提取的数量，该类不支持稀疏矩阵输入

#低秩近似的类
def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
    
    A = np.array(
        [
            [3, 2, -2, -3, 2, 21, 3, 4],
            [2, 4, -1, -7, 4, 45, 4, 4],
            [2, 3, 41, 5, 23, -4, 6, 4],
            [5, 7, 20, 6, 30, -7, 6, 9]
        ]
    )

    # p为近似矩阵的秩，秩p<=r
    p = 2
    B = approximation(A, p)
    print(B)
    #可以看到最终得到的矩阵秩为2
    print(np.linalg.matrix_rank(B),"\n\n")
    
    #取矩阵秩为4
    p = 4
    B = approximation(A, p)
    print(B)
    print(np.linalg.matrix_rank(B))

    # 调用api核对，和传统PCA比较
    # pca= PCA(n_components=2, svd_solver='auto')
    # B2 = pca.fit_transform(A)
    # print(B2)


# # 尝试单张图片的SVD近似

# In[23]:


import numpy as np
from sklearn.decomposition import PCA
import cv2
#from PIL import Image
import matplotlib.pyplot as plt

#rc配置
plt.rcParams[u'font.sans-serif'] = ['simhei'] #图形中的中文正常编码显示
plt.rcParams['axes.unicode_minus'] = False #刻度正常显示正负号

def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
    #img_PIL = Image.open("G:\Myself\大创-几何重构&性能预测\矩阵降维\SVD低秩逼近\code\result.jpeg")
    img = cv2.imread("G:/Myself/大创-几何重构&性能预测/矩阵降维/SVD低秩逼近/code/img.jpeg", flags=0)
    img_output = img.copy()

    # p为近似矩阵的秩，秩p<=r，p越大图像压缩程度越小，越清晰
    p = 50
    img_output = approximation(img, p)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[1].imshow(img_output)
    axs[1].set_title('压缩后的图')
    plt.savefig('G:/Myself/大创-几何重构&性能预测/矩阵降维/SVD低秩逼近/code/result.jpeg')
    plt.show()


# # 对dpi_64单张进行低秩近似 

# In[25]:


import numpy as np
from sklearn.decomposition import PCA 
#decomposition随机截断的SVD,取决于输入数据的形状和要提取的数量，该类不支持稀疏矩阵输入
#读取处理xlsx文件
from openpyxl import load_workbook

#低秩近似的类
def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
    
    A = numpy.loadtxt(open("G:/Myself/大创-几何重构&性能预测/dpi_10000/dpi_64_10000/1.CSV","rb"), delimiter=",",skiprows=0)
    
    # p为近似矩阵的秩，秩p<=r
    p = 20
    B = approximation(A, p)
    print(B)
    #可以看到最终得到的矩阵秩为20
    print(np.linalg.matrix_rank(B),"\n\n")
    
    # p为近似矩阵的秩，秩p<=r
    p = 45
    B = approximation(A, p)
    print(B)
    #可以看到最终得到的矩阵秩为26(说明A的秩为26)
    print(np.linalg.matrix_rank(B),"\n\n")


# In[ ]:




