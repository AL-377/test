import os
import pandas as pd
import torch

'''
    创建数据文件
    填写文件
    读取文件
    处理数据，转化为向量

'''

# os.path.join()函数：连接两个或更多的路径名组件
#
#                          1.如果各组件名首字母不包含’/’，则函数会自动加上
#
# 　　　　　　　　　2.如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
#
# 　　　　　　　　　3.如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾

def createFolder():
    os.makedirs(os.path.join('data'), exist_ok=True)
    data_file = os.path.join('data', 'house_tiny1.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    return data_file

def readData(data_file):
    data = pd.read_csv(data_file)
    # print(data)
    return data

def data2vec(data):
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())  #填入缺项值
    # print(inputs)
    inputs = pd.get_dummies(inputs, dummy_na=True) #按照类别填0 1，可以指定不同属性分类
    print(type(inputs))

    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    #前两列为输入，最后为输出。
    return X,y

if __name__ == '__main__':
    datafile=createFolder()
    data=readData(datafile)
    vec_x,vec_y=data2vec(data)
    print(vec_x,vec_y)
