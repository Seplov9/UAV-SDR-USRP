# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:26:29 2023

@author: cyk
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

a=0

while a<=0.8:

  data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240409/R/3.1G-4M-70m/R_c-D-S/data.mat')
  # data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240308/150/input.1/data_to_real.mat')
  X = data['X']
  y = data['Y'][0]

  # 数据集切分
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(y_test)

  # 数据标准化
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # 创建决策树模型
  # clf = DecisionTreeClassifier()
  clf = DecisionTreeClassifier(criterion='gini',       # 使用Gini不纯度
                             max_depth=3,            # 最大深度为3
                             min_samples_split=4,    # 节点分裂最小样本数为4
                             min_samples_leaf=2)     # 叶节点最小样本数为2

  # 训练模型
  clf.fit(X_train, y_train)

  # 使用模型进行预测
  y_pred = clf.predict(X_test)

  a=accuracy_score(y_test, y_pred)
  print(a)

# 打印预测结果及模型评分
print("Accuracy score: ", accuracy_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred,average='micro'))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # confusion matrix
 
# confusion= confusion_matrix(y_test, y_pred)
 
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
# plt.title('KNN Confusion Matrix')
 
# # plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
 
# # 显示数据
# for first_index in range(len(confusion)):  # 第几行
#     for second_index in range(len(confusion[first_index])):  # 第几列
#         plt.text(first_index, second_index, confusion[first_index][second_index])
# # 在matlab里面可以对矩阵直接imagesc(confusion)
 
# # # 显示
# plt.show()

# # confusion matrix-2

# confusion= confusion_matrix(y_test, y_pred)

# # 计算百分比
# confusion_percentage = confusion / confusion.sum(axis=1, keepdims=True)

# # 使用浅色的颜色图
# plt.imshow(confusion_percentage, cmap=plt.cm.Blues, alpha=0.7)  # alpha控制透明度，使颜色变浅

# # 坐标轴设置
# indices = range(len(confusion))
# # plt.xticks(indices, ['Detection', 'U-Stable', 'L-U-Stable' ,'U-Moving','L-U-Moving', 'Noise'])
# # plt.xticks(indices, ['Detection', 'U-Stable', 'U-Moving', 'Noise'])
# # plt.yticks(indices, ['Detection', 'U-Stable', 'U-Moving', 'Noise'])
# # plt.xticks(indices, ['Detection', 'L-U-Stable', 'L-U-Moving', 'Noise'])
# # plt.yticks(indices, ['Detection', 'L-U-Stable', 'L-U-Moving', 'Noise'])
# plt.xticks(indices, ['没有无人机', '有无人机'])
# plt.yticks(indices, ['没有无人机', '有无人机'])

# plt.colorbar()

# plt.xlabel('Pred Labels')
# plt.ylabel('True Labels')
# # plt.title('KNN Confusion Matrix')
# plt.title('DT Confusion Matrix')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
# plt.rcParams['axes.unicode_minus'] = False

# # 显示数据（转换为百分比）
# for i in range(len(confusion_percentage)):
#     for j in range(len(confusion_percentage[i])):
#         plt.text(j, i, f"{confusion_percentage[i][j]*100:.1f}%", ha="center", va="center", color="black")

# plt.show()
