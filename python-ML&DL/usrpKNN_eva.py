# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:26:29 2023

@author: cyk
"""
import numpy as np
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
from sklearn.model_selection import cross_val_score, KFold
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

a=0

while a<=0:

# 加载mat文件中的数据
  mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240329/3.1G-4M-70m/data-cfr/data.mat')
  # mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240409/R/3.1G-4M-70m/R_c-D-S/data.mat')
# mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240319/1.6G/D-S/R_c.mat')

  X = mat_data['X']
  Y = mat_data['Y'][0]
  print(X.shape)
  print(Y.shape)
# Y = mat_data['Y'].squeeze()  # 将Y转换为一维数组

  indices_to_keep = [0, 2, 3, 4, 5, 6]
# X_modified = X[:, indices_to_keep]

# X = np.clip(X, a_min=-1e10, a_max=1e10)

# 数据集切分
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(X_modified, Y, test_size=0.3, random_state=42)

# 数据标准化
  scaler = StandardScaler()
# X = scaler.fit_transform(X)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

# 创建KNN模型，设置邻居数为3
  knn = KNeighborsClassifier(n_neighbors=3)
  # knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# 训练模型
  knn.fit(X_train, y_train)

# 进行预测
  y_pred = knn.predict(X_test)

  a=accuracy_score(y_test, y_pred)
  print(a)

# 打印预测结果及模型评分
print("Accuracy score: ", accuracy_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred,average='micro'))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# # 定义多折交叉验证的参数
# kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 使用5折交叉验证

# # 执行多折交叉验证并评估模型
# scores = cross_val_score(knn, X, Y, cv=kf)

# # 输出交叉验证结果
# print("Cross-validation scores:", scores)
# print("Mean accuracy:", scores.mean())
# print("Standard deviation:", scores.std())

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

# # plt.rcParams.update({'font.size': 12})
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})
# plt.rcParams['pdf.fonttype'] = 42

# confusion= confusion_matrix(y_test, y_pred)

# # 计算百分比
# confusion_percentage = confusion / confusion.sum(axis=1, keepdims=True)

# # 使用浅色的颜色图
# plt.imshow(confusion_percentage, cmap=plt.cm.Blues, alpha=0.7)  # alpha控制透明度，使颜色变浅

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
# plt.title('KNN Confusion Matrix')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
# plt.rcParams['axes.unicode_minus'] = False

# # 显示数据（转换为百分比）
# for i in range(len(confusion_percentage)):
#     for j in range(len(confusion_percentage[i])):
#         plt.text(j, i, f"{confusion_percentage[i][j]*100:.1f}%", ha="center", va="center", color="black")

# # 保存为PDF文件
# plt.tight_layout()
# # plt.savefig('picture/KNN.pdf', format='pdf', dpi=300)

# plt.show()
