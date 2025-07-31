# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:37:07 2023

@author: cyk
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

a=0

while a<=0:

# 加载mat文件中的数据
  # mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240319/1.6G/input.1/data_to_real.mat')
  # mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240308/150/input.1/data_to_real.mat')
  mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240329/3.1G-4M-70m/data-cfr/data.mat')

  X = mat_data['X']
  Y = mat_data['Y'].squeeze()  # 将Y转换为一维数组

  indices_to_keep = [0, 2, 3, 4, 5, 6]
# X_modified = X[:, indices_to_keep]

# 数据集切分
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X_modified, Y, test_size=0.3, random_state=42)

# 数据标准化
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

# # SVM分类器模型训练
# # svm_model = SVC(kernel='linear', C=1.0)
# svm_model = SVC(kernel='rbf', C=1.0)
# svm_model.fit(X_train, y_train)
# # 预测测试数据集
# predicted_y = svm_model.predict(X_test)

# 创建MLP分类器
  mlp = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)
# 训练模型
  mlp.fit(X_train, y_train)
# 预测
  predicted_y = mlp.predict(X_test)

  a=accuracy_score(y_test, predicted_y)
  print(a)

# 打印预测结果及模型评分
# print("Predicted labels: ", predicted_y)
# print("Accuracy score: ", svm_model.score(X_test, y_test))
print("Accuracy score: ", accuracy_score(y_test, predicted_y))
# print("F1 Score:", f1_score(y_test, predicted_y))
# print("F1 Score:", f1_score(y_test, predicted_y, average='micro'))

# # confusion matrix-1
 
# confusion= confusion_matrix(y_test, predicted_y)
 
# plt.imshow(confusion, cmap=plt.cm.Blues)
 
# # 热度图，后面是指定的颜色块，可设置其他的不同颜色
# plt.imshow(confusion, cmap=plt.cm.Blues)
# # ticks 坐标轴的坐标点
# # label 坐标轴标签说明
# indices = range(len(confusion))
# # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# # plt.xticks(indices, [0, 1, 2])
# # plt.yticks(indices, [0, 1, 2])
# plt.xticks(indices, ['detection', 'P', 'UAV', 'N'])
# plt.yticks(indices, ['detection', 'P', 'UAV', 'N'])
 
# plt.colorbar()
 
# plt.xlabel('True Labels')
# plt.ylabel('Predicted Labels')
# plt.title('SVM Confusion Matrix')
 
# # plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
 
# # 显示数据
# for first_index in range(len(confusion)):  # 第几行
#     for second_index in range(len(confusion[first_index])):  # 第几列
#         plt.text(first_index, second_index, confusion[first_index][second_index])
# # 在matlab里面可以对矩阵直接imagesc(confusion)
 
# # 显示
# plt.show()

# # confusion matrix-2

# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})
# plt.rcParams['pdf.fonttype'] = 42

# confusion = confusion_matrix(y_test, predicted_y)

# # 计算百分比
# confusion_percentage = confusion / confusion.sum(axis=1, keepdims=True)

# # 使用浅色的颜色图
# plt.imshow(confusion_percentage, cmap=plt.cm.Blues, alpha=0.7)

# # 坐标轴设置
# indices = range(len(confusion))
# # plt.xticks(indices, ['Detection', 'U-Stable', 'L-U-Stable' ,'U-Moving','L-U-Moving', 'Noise'])
# plt.xticks(indices, ['Detection', 'Stable', 'Moving', 'Noise'])
# plt.yticks(indices, ['Detection', 'Stable', 'Moving', 'Noise'])
# # plt.xticks(indices, ['Detection', 'L-U-Stable', 'L-U-Moving', 'Noise'])
# # plt.yticks(indices, ['Detection', 'L-U-Stable', 'L-U-Moving', 'Noise'])
# # plt.xticks(indices, ['没有无人机', '有无人机'])
# # plt.yticks(indices, ['没有无人机', '有无人机'])

# plt.colorbar()

# plt.xlabel('Pred Labels')
# plt.ylabel('True Labels')
# plt.title('MLP Confusion Matrix')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
# plt.rcParams['axes.unicode_minus'] = False

# # 显示数据（转换为百分比）
# for i in range(len(confusion_percentage)):
#     for j in range(len(confusion_percentage[i])):
#         plt.text(j, i, f"{confusion_percentage[i][j]*100:.1f}%", ha="center", va="center", color="black")

# # 保存为PDF文件
# # plt.savefig('picture/MLP.pdf', format='pdf', dpi=300)

# plt.show()