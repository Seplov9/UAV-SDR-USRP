import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io

# 加载MAT文件数据
# file_path = 'C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240329/3.1G-1M/input.1/data_to_real.mat'
file_path = 'C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240409/R/1.6G-4M/conj-0.5/D/data.mat'
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
Y = mat_data['Y'].flatten()  # 转换为一维数组

indices_to_keep = [0, 2, 3, 4, 5, 6]
X_modified = X[:, indices_to_keep]

# 数据集切分
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_modified, Y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换标签为one-hot编码
num_classes = len(np.unique(y_train))+1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 多头自注意力层
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # 前馈网络
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# 输入层
inputs = Input(shape=(4800,))

# 增加维度匹配Transformer输入要求
x = Dense(64, activation="relu")(inputs)
x = Reshape((8, 8))(x)  # 使用适当的Reshape以匹配Transformer的输入

# Transformer编码器
x = transformer_encoder(x, head_size=8, num_heads=2, ff_dim=128, dropout=0.1)

# 展平并连接全连接层
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
# outputs = Dense(activation="softmax")(x)
outputs = Dense(num_classes, activation="softmax")(x)

# 创建和编译模型
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 评估模型
model.evaluate(X_test, y_test)
