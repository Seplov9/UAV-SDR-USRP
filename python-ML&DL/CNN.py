import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# 加载数据
mat_data = loadmat('C:/Users/cyk/Desktop/Bluetooth/usrp231210/0db/usrp240329/3.1G-4M-70m/data-cfr/data.mat')
X = mat_data['X']
Y = mat_data['Y'][0]
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加 channel 维度
Y = torch.tensor(Y, dtype=torch.long).squeeze()  # 去除多余维度

# 自动获取类别数
num_classes = len(torch.unique(Y))  # 自动检测类别数量

# 使用train_test_split来划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将训练集和测试集转换为TensorDataset
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义1D CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        # 计算经过卷积和池化操作后的特征图尺寸
        self.fc1_input_dim = self._get_fc1_input_dim(input_dim)
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 输出为动态类别数

    def _get_fc1_input_dim(self, input_length):
        # 通过一个虚拟的输入计算经过卷积和池化操作后的特征图大小
        x = torch.zeros(1, 1, input_length)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.fc1_input_dim)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 获取输入特征维度
input_dim = X.shape[2]  # 输入维度（每个样本的特征数）

# 实例化模型
model = CNN1D(input_dim=input_dim, num_classes=num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 训练模型
# def train_model(model, train_loader, criterion, optimizer, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

# # 评估模型
# def evaluate_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     print(f'Test Accuracy: {accuracy:.4f}')

# # 训练和评估
#     (model, train_loader, criterion, optimizer, epochs=10)
# evaluate_model(model, test_loader)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        # 模型设为训练模式
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 训练集上训练
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training', unit='batch'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        
        # 模型设为评估模式
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        # 在测试集上验证
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} - Validating', unit='batch'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_accuracy = correct_test / total_test
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5)
