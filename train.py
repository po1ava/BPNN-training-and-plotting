import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, ELU
from tensorflow.keras import optimizers
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 确定随机数，确保训练结果可重复
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

file_path = 'example_data.xlsx'  # 读取数据的表格路径
input_column = [1, 3, 5, 7, 9]  # 模型输入项所在的表格列数组
output_column = [10]  # 模型输出项所在的表格列数组

units_min = 5  # 训练中测试的每层最小神经元数量
units_max = 25  # 训练中测试的每层最大神经元数量(包含)
units_interval = 5  # 训练中测试的神经元间隔数量
hidden_layer_num = 3  # 模型中使用的隐藏层数量
epoch_num = 1000  # 模型的最大训练步数
log_length = 5  # 记录一次R²数据的模型训练步数间隔

learning_rate = 0.005  # 模型学习率
loss_function = 'mse'  # 模型使用的损失函数
optimizer = optimizers.Adam(learning_rate=learning_rate)  # 模型使用的优化器
activation = 'relu'  # 除输出层外，其余层使用的激活函数
activation_output = activation  # 模型输出层使用的激活函数
scaler = MinMaxScaler()  # 预处理时使用的数据归一化方法
k_splits = 5  # k-fold划分数量，也决定训练与测试集划分比例
k_splits_on = True  # 是否执行k-fold交叉验证多次训练
patience = 100000  # 验证集性能持续未改善时停止训练的连续记录次数

log_name = f'{activation}_l{hidden_layer_num}_lr{learning_rate}'  # 训练日志输出名称
model_name = log_name  # 保存的最优模型文件输出名称

if __name__ == '__main__':

    log_num = int(epoch_num / log_length)
    units_test_num = int((units_max - units_min)/units_interval)+1
    input_num = len(input_column)
    output_num = len(output_column)

    x = pd.read_excel(file_path).iloc[:, input_column]  # 读取模型输入项
    y = pd.read_excel(file_path).iloc[:, output_column]  # 读取模型输出项

    x = scaler.fit_transform(x)
    y = scaler.fit_transform(pd.DataFrame(y))

    # 创建输出数据储存字典与输出表格框架
    r2 = {'name': [], 'model_r2': [], 'r2_pos': []}
    for i in range(units_min, units_max+1, units_interval):  # 初始化三列基本信息
        for j in range(1, 6):
            r2['name'].append(f'units{i}_{j}')
            r2['model_r2'].append(-10)
            r2['r2_pos'].append(-10)
    for i in range(1, log_num + 1):  # 填充训练集r2列
        temp = []
        for j in range(units_test_num * 5):
            temp.append(-10)
        r2[f'train{int(i * log_length)}'] = temp
    for i in range(1, log_num + 1):  # 填充测试集r2列
        temp = []
        for j in range(units_test_num * 5):
            temp.append(-10)
        r2[f'val{int(i * log_length)}'] = temp

    # 开始训练，根据隐藏层单元数划分
    for units in range(units_min, units_max+1):

        # K-fold划分训练集与测试集
        kf = KFold(n_splits=k_splits, shuffle=True, random_state=0)
        k = 0  # 用于在输出时记录k-fold训练进度

        # 根据K-fold划分训练与验证集
        for train, test in kf.split(x):

            if (not k_splits_on) and k != 0:  # 不执行交叉验证时退出训练
                continue

            x_train = np.array(x)[train]
            x_test = np.array(x)[test]
            y_train = np.array(y)[train]
            y_test = np.array(y)[test]

            max_test = -10  # 最大测试集r2值
            max_pos = -10  # 测试集r2最大值时的训练位置

            # 搭建模型结构
            model = Sequential()
            model.add(Dense(units=units, input_dim=input_num, activation=activation))
            for i in range(hidden_layer_num - 1):
                model.add(Dense(units=units, activation=activation))
            model.add(Dense(units=output_num, activation=activation_output))
            model.compile(loss=loss_function, optimizer=optimizer)
            model.summary()

            # 开始模型训练
            for i in range(1, log_num + 1):
                model.fit(x=x_train, y=y_train, epochs=log_length, batch_size=256)
                r2_train = r2_score(y_train, model.predict(x_train))
                r2_test = r2_score(y_test, model.predict(x_test))

                if r2_test > max_test:
                    max_pos = i
                    max_test = r2_test

                row_num=int((units - units_min)/units_interval) * 5 + k

                r2[f'train{int(i * log_length)}'][row_num] = r2_train
                r2[f'val{int(i * log_length)}'][row_num] = r2_test

                # 模型训练早停机制
                if i - max_pos > patience or i == log_num:
                    r2['model_r2'][row_num] = max_test
                    r2['r2_pos'][row_num] = max_pos
                    k += 1

                    model.save(f'{model_name}.h5')
                    print(f'r2={max_test}')
                    break

    r2 = pd.DataFrame(r2)
    r2.to_excel(f'{log_name}.xlsx', index=False)
