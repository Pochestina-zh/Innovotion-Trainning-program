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

# In[11]:


import numpy as np
from sklearn.decomposition import PCA 
#decomposition随机截断的SVD,取决于输入数据的形状和要提取的数量，该类不支持稀疏矩阵输入

#低秩近似的类
def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print(s)
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

# In[5]:


import numpy as np
from sklearn.decomposition import PCA
import cv2
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
    
    img = cv2.imread('G:/Myself/work1_matrix/SEM.jpg',flags=0)
    img_output = img.copy()
    
    # p为近似矩阵的秩，秩p<=r，p越大图像压缩程度越小，越清晰
    p = 6
    img_output = approximation(img, p)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[1].imshow(img_output)
    axs[1].set_title('压缩后的图')
    plt.savefig('G:/Myself/work1_matrix/result1_SEM.jpeg')
    plt.show()
    
    p = 15
    img_output = approximation(img, p)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[1].imshow(img_output)
    axs[1].set_title('压缩后的图')
    plt.savefig('G:/Myself/work1_matrix/result2_SEM.jpeg')
    plt.show()
    
    p = 50
    img_output = approximation(img, p)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('原图')
    axs[1].imshow(img_output)
    axs[1].set_title('压缩后的图')
    plt.savefig('G:/Myself/work1_matrix/result3_SEM.jpeg')
    plt.show()


# # 对dpi_64、dpi_512单张进行低秩近似 

# In[17]:


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
## dpi_64
    print("（一）dpi_64 as follows:\n")
    A = numpy.loadtxt(open("G:/Myself/大创-几何重构&性能预测/dpi_10000/dpi_64_10000/1.csv","rb"), delimiter=",",skiprows=0)
    
    # p为近似矩阵的秩，秩p<=r
    p = 20
    B = approximation(A, p)
    print("dpi_64_1 ,p=20\n", B)
    #可以看到最终得到的矩阵秩为20
    print("rank:", np.linalg.matrix_rank(B),"\n")
    
    # p为近似矩阵的秩，秩p<=r
    p = 45
    B = approximation(A, p)
    print("dpi_64_1 ,p=45\n",B)
    #可以看到最终得到的矩阵秩为26(说明A的秩为26)
    print("rank:", np.linalg.matrix_rank(B),"\n\n")
    
## dpi_512
    print("（二）dpi_512 as follows:\n")
    A = numpy.loadtxt(open("G:/Myself/大创-几何重构&性能预测/dpi_10000/dpi_512_10000/1.csv","rb"), delimiter=",",skiprows=0)
    
    # p为近似矩阵的秩，秩p<=r
    p = 50
    B = approximation(A, p)
    print("dpi_512_1 ,p=20\n", B)
    #可以看到最终得到的矩阵秩为20
    print("rank:", np.linalg.matrix_rank(B),"\n")
    


# # 选取%85奇异值低秩近似

# In[1]:


import numpy as np
from sklearn.decomposition import PCA 
#decomposition随机截断的SVD,取决于输入数据的形状和要提取的数量，该类不支持稀疏矩阵输入
#读取处理xlsx文件
#from openpyxl import load_workbook

#低秩近似的类
def approximation(A):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print("奇异值总和:",sum(s))  #总奇异值
    #选取85%的奇异值
    total_s = 0
    for i in range(len(s)):
        total_s += s[i]
        if total_s >= 0.9 * sum(s):
            p = i
            print("用到的奇异值个数：",p)
            break
    #计算近似矩阵
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

if __name__ == '__main__':
## dpi_64
    print("dpi_64 as follows:\n")
    A = np.loadtxt(open("G:/Myself/大创-几何重构&性能预测/dpi_10000/dpi_64_10000/1.csv","rb"), delimiter=",",skiprows=0)# 对csv文件进行处理
    B = approximation(A)
    print(A)
    print("dpi_64_1:\n", B)
    


# In[2]:


import numpy as np
from sklearn.decomposition import PCA 
#decomposition随机截断的SVD,取决于输入数据的形状和要提取的数量，该类不支持稀疏矩阵输入
#读取处理xlsx文件
import pandas as pd


#低秩近似的类
def approximation(A):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print("奇异值总和:",sum(s))  #总奇异值
    #选取85%的奇异值
    total_s = 0
    for p in range(len(s)):
        total_s += s[p]
        if total_s >= 0.9 * sum(s):
            print("用到的奇异值个数：",p)
            break
    #计算近似矩阵
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B


if __name__ == '__main__':
## dpi_64
    print("dpi_64 as follows:\n")
    A = pd.read_excel("G:/Myself/大创-几何重构&性能预测/dpi_10000/dpi_64_10000/1.xlsx",header = None)#对xlsx文件进行处理
    B = approximation(A)
    print(A)
    print("dpi_64_1:\n", B)


# # 批量处理xlsx文件dpi_64

# In[32]:


import pandas as pd
import os
import numpy
# 全局变量，文件读取路径
read_path = ""
# 全局变量，处理结果文件输出路径
output_path = ""


#低秩近似的类
def approximation(A):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print("奇异值总和:",sum(s))  #总奇异值
    #选取85%的奇异值
    total_s = 0
    for p in range(len(s)):
        total_s += s[p]
        if total_s >= 0.85 * sum(s):
            print("用到的奇异值个数：",p)
            break
    #计算近似矩阵
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B


# 获取文件路径
def get_file_path():
    read_path = r"G:\Myself\大创-几何重构&性能预测\dpi_10000 - 副本(未处理)\dpi_64_10000"
    output_path = r"G:\Myself\大创-几何重构&性能预测\dealed\dealed_64"
    return read_path,output_path


# 读取文件名称和内容
def deal_files():
    # 获取read_path下的所有文件名称（顺序读取的）
    files = os.listdir(read_path)
    for file_name in files:
        # 读取单个文件内容(单个矩阵)
        A = pd.read_excel(read_path+"\\"+file_name, header = None)
        #处理单个文件(调用低秩近似)
        B = approximation(A)
        # 输出结果到指定路径下
        pd.DataFrame(B).to_excel(excel_writer = output_path + "\\" + "处理结果_" + file_name, index=False, header = False)
    print("文件处理完毕")



# 主函数
if __name__=="__main__":
    # 获取文件输入和输出路径
    read_path,output_path = get_file_path()
    # 开始处理文件，并输出处理文件结果
    deal_files()


# # 压缩单张low_contrast.2950.png

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import pandas as pd
from PIL import Image

##################存Up,sp,Vp的类################
def save(U,s,V_T,p):
    Up = U[:p,:]
    sp = s[:p]
    V_Tp = V_T[:,:p]
    pd.DataFrame(Up).to_excel(r"G:\Myself\大创-几何重构&性能预测\compressed\test\Up2950.xlsx",index = False, header = None)
    pd.DataFrame(sp).to_excel(r"G:\Myself\大创-几何重构&性能预测\compressed\test\sp2950.xlsx",index = False, header = None)
    pd.DataFrame(V_Tp).to_excel(r"G:\Myself\大创-几何重构&性能预测\compressed\test\VTp2950.xlsx",index = False, header = None)
    pass

##################低秩近似的类################
def approximation(A):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print("奇异值总和:",sum(s))  #总奇异值
    #选取%的奇异值
    total_s = 0
    for p in range(len(s)):
        total_s += s[p]
        if total_s >= 0.6 * sum(s):
            print("用到的奇异值个数：",p,"\n")
            #save(U,s,V_T,p)
            break
    #计算近似矩阵
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

#################图片读取处理#################
if __name__ == '__main__':
    img = mpimg.imread(r"G:\Myself\大创-几何重构&性能预测\3000_low constrast\3000\2950.png")
    print("原始的图片格式:", img.shape,"\n")

#创建3个和图片大小相同的0矩阵
    A = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)

#复制图像通道里的数据成二维tensor
    A[:,:] = img[:,:,0]  # 复制 b 通道的数据
    print("读取的图片数据:\n",A,"\n")
###################图片压缩#####################
    B = approximation(A)
    print("压缩之后的图片数据:\n", B,"\n")

##################图片保存######################
    outputimg = Image.fromarray(B * 255.0)
    outputimg = outputimg.convert('L')
    outputimg.save(r"G:\Myself\大创-几何重构&性能预测\SVD_compressed\test\test2_2950.png")  


# # 批量压缩low_contrast的png文件

# In[55]:


import pandas as pd
import os
import numpy

# 全局变量，文件读取路径
read_path = ""
# 全局变量，处理结果文件输出路径
output_path = ""

# 获取文件路径
def get_file_path():
    read_path = "G:/Myself/大创-几何重构&性能预测/3000_low constrast/3000"
    output_path = "G:/Myself/大创-几何重构&性能预测/compressed"
    return read_path,output_path


# 读取文件名称和内容
def deal_files():
    # 获取read_path下的所有文件名称（顺序读取的）
    files = os.listdir(read_path)
    for file_name in files:
        # 读取单个文件内容(单个矩阵)
        img = mpimg.imread(read_path+"\\"+file_name)
        A = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
        A[:,:] = img[:,:,0]
        #处理单个文件(调用低秩近似)
        B = approximation(A,file_name)
        # 输出结果到指定路径下
        outputimg = Image.fromarray(B * 255.0)
        outputimg = outputimg.convert('L')
        outputimg.save(output_path + "\\" + "Compressed_" + file_name)
    print("文件处理完毕")
    
##################存Up,sp,Vp的类################
def save(U,s,V_T,p,file_name):
    Up = U[:p,:]
    sp = s[:p]
    V_Tp = V_T[:,:p]
    pd.DataFrame(Up).to_excel(excel_writer = output_path+"/"+"matrix"+"/"+"U_"+file_name[0:-3] + "xlsx",index = False, header = None)
    pd.DataFrame(sp).to_excel(excel_writer =output_path+"/"+"matrix"+"/"+"s_"+file_name[0:-3] + "xlsx",index = False, header = None)
    pd.DataFrame(V_Tp).to_excel(excel_writer =output_path+"/"+"matrix"+"/"+"VT_"+file_name[0:-3] + "xlsx",index = False, header = None)
    pass

##################低秩近似的类##################
def approximation(A, file_name):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    print("奇异值总和:",sum(s))  #总奇异值
    #选取90%的奇异值
    total_s = 0
    for p in range(len(s)):
        total_s += s[p]
        if total_s >= 0.9 * sum(s):
            print("用到的奇异值个数：",p)
            save(U,s,V_T,p,file_name)
            break
    #计算近似矩阵
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

######################主函数########################
if __name__=="__main__":
    # 获取文件输入和输出路径
    read_path,output_path = get_file_path()
    # 开始处理文件，并输出处理文件结果
    deal_files()
    


