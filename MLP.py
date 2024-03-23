import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score
from sklearn.linear_model import Ridge
import copy
import torch
import torch.nn.functional as F
import copy
from torch import nn, optim
import torch.utils.data as Data
import time
import math
def read(path):
    df = pd.read_excel(path,index_col=0)
    df.columns = np.arange(len(df.loc['U_8126464',:]))
    df.index = np.arange(len(df))
    return df
#data是交互次数矩阵，data01是是否交互的矩阵
data = read('user_course_交互次数.xlsx')
data01 = read('user_course_matrix.xlsx')
#随机选取一些交互次数来当x，一些是否交互当y
np.random.seed(2023)
user_mask = np.arange(len(data))
np.random.shuffle(user_mask)
cour_mask = np.arange(len(data.loc[0,:]))
np.random.shuffle(cour_mask)
x_train,y_train = data.loc[user_mask[:1500],cour_mask[:250]],data01.loc[user_mask[:1500],cour_mask[250:]]
x_test,y_test = data.loc[user_mask[1500:],cour_mask[:250]],data01.loc[user_mask[1500:],cour_mask[250:]]
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
y_train_copy = copy.deepcopy(y_train)
y_test_copy = copy.deepcopy(y_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
#将数据类型转换为tensor方便pytorch使用
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


# 搭建全连接神经网络回归
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # 第一个隐含层
        self.hidden1 = nn.Linear(in_features=250, out_features=50, bias=False)
        # 第二个隐含层
        #         self.hidden2 = nn.Linear(100, 100)
        # 第三个隐含层
        #         self.hidden3 = nn.Linear(100, 50)
        # 回归预测层
        self.predict = nn.Linear(50, 1)

    # 定义网络前向传播路径
    def forward(self, x):
        x = F.sigmoid(self.hidden1(x))
        #         x = F.relu(self.hidden2(x))
        #         x = F.relu(self.hidden3(x))
        #         out=F.softmax(self.predict(x),dim=1) #输出层采用softmax函数
        out = self.predict(x)
        # 输出一个一维向量
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testnet = MLPregression().to(device)
# 将训练数据处理为数据加载器
y_predicts = []
y_tests = []
x_test = x_test.to(device)
for j in range(y_train.shape[1]):
    start = time.time()
    train_data = Data.TensorDataset(x_train, y_train[:,j])
    test_data = Data.TensorDataset(x_test, y_test[:,j])
    train_loader = Data.DataLoader(dataset = train_data, batch_size = 64,
                                   shuffle = True, num_workers = 1)
    # 定义优化器
    optimizer = torch.optim.SGD(testnet.parameters(), lr = 0.01)
    loss_func = nn.MSELoss().to(device) # 均方根误差损失函数


    # 对模型迭代训练，总共epoch轮
    for epoch in range(30):
        train_loss = 0
        train_num = 0
        # 对训练数据的加载器进行迭代计算
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x,b_y = b_x.to(device),b_y.to(device)
            output = testnet(b_x) # MLP在训练batch上的输出
            loss = loss_func(output, b_y) # 均方根损失函数
            optimizer.zero_grad() # 每次迭代梯度初始化0
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 使用梯度进行优化
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
    y_pre = testnet(x_test)
    y_pre = y_pre.cpu()
    y_pre = y_pre.data.numpy()
    y_predicts.append(list(y_pre))
    y_tests.append(list(np.array(y_test[:,j])))
    print(j)
y_predicts = np.array(y_predicts).T
y_tests = np.array(y_tests).T