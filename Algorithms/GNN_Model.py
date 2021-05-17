import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

## 数据排序
def sort_data(data, return_order=False):
    N = data.shape[0]
    # 计算距离
    distance_matrix = distance.cdist(data, data, 'euclidean')
    distance_matrix[np.where(distance_matrix < 1e-12)] = np.inf
    # print(distance_matrix)
    # 找出距离最近的两个点
    min_index = np.argmin(distance_matrix)
    min_index = [min_index//N, min_index%N]
    # 开始排序
    remain = list(range(N))
    order = [min_index[0], min_index[1]]
    remain.remove(order[0])
    remain.remove(order[-1])
    while(len(remain) != 0):
        # print(distance_matrix)
        # 找到与当前order0头或尾最近的点
        min_index_A = np.argmin(distance_matrix[order[0], remain])
        min_index_B = np.argmin(distance_matrix[order[-1], remain])
        if distance_matrix[order[0], remain[min_index_A]] <= distance_matrix[order[-1], remain[min_index_B]]:
            order = [remain[min_index_A]] + order
            remain.remove(remain[min_index_A])
        else:
            order = order + [remain[min_index_B]]
            remain.remove(remain[min_index_B])
    
    # 判断是否需要反转
    if np.sum(data[order[0], :]**2 > data[order[-1], :]**2):
        order = order[: :-1]
    sorted_data = data[order, :]
    if return_order:
        return order
    else:
        return sorted_data


## 聚类
def data_cluster(data):
    """
    数据聚类。
    方法：先用DBScan，如果发现聚类数目过多，则采用k-means
    """

    distance_matrix = distance.cdist(data, data, 'euclidean')

    cluster = DBSCAN(eps = np.mean(distance_matrix))
    y_pred = cluster.fit_predict(data)
    cluster_num = len(np.unique(y_pred))
    if cluster_num > data.shape[0] // 4:
        cluster = KMeans(n_clusters=2)
        y_pred = cluster.fit_predict(data)

    unique_cluster_num = np.unique(y_pred)

    raw_data_set = []
    cluster_centers = []
    for i in range(len(unique_cluster_num)):
        cluster_data = data[np.where(y_pred == unique_cluster_num[i])[0], :]
        raw_data_set.append(cluster_data)
        cluster_centers.append(np.mean(cluster_data, axis=0))
    
    # 聚类排序
    cluster_centers = np.vstack(cluster_centers)
    # 数据排序
    if cluster_centers.shape[0] > 1:
        order = sort_data(cluster_centers, return_order=True)
        data_set = []
        for i in range(len(order)):
            data_set.append(raw_data_set[i])
    else:
        data_set = raw_data_set

    return data_set


## 结点网络
class NodeNet(nn.Module):
    def __init__(self, Dim, hidden_dim=1000):
        """
        子网络：拟合隐变量与各个数据的关系
        """
        super(NodeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, Dim, bias=True),
        )
        self.Dim = Dim
        self.loss = None  # 训练过程中实时地存储loss
        self.loss_list = []  # 用于记录训练过程的loss value
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.outputParameter_index_list = list(range(self.Dim))
        self.raw_dim = None  # 记录原本的output dim
    
    def forward(self, x):
        x = self.model(x)
        return x


def train(data_set, epochs=1000, hidden_dim=1000):
    """
    注意：传入的data_set是聚类后的数据,已经归一化,无需再归一化.这里只需对data_set的每一类数据再进行排序即可
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # 存储所有的网络输出数据
    total_predicts_data = []

    # 创建网络
    network_list = []
    
    # 网络准备
    for k in range(len(data_set)):
        data = data_set[k]
        raw_dim = data.shape[1]
        outputParameter_index_list = list(range(raw_dim))
        
        # 精简数据
        for i in range(raw_dim):
            if np.std(data[:, [i]]) <= 1e-2:
                outputParameter_index_list.remove(i)
        data_set[k] = (data_set[k])[:, outputParameter_index_list]
        Dim = len(outputParameter_index_list)  # 更新Dim
        nodeNet = NodeNet(Dim, hidden_dim).to(device)
        nodeNet.raw_dim = raw_dim
        nodeNet.outputParameter_index_list = outputParameter_index_list
        network_list.append(nodeNet)
        
    # 训练网络
    for epoch in range(1, epochs+1):
        # 前馈
        total_predicts_data = []
        for k in range(len(data_set)):
            data = data_set[k]
            # 数据排序
            if data.shape[0] > 1:
                data = sort_data(data)
            data = torch.as_tensor(data, dtype=torch.float32).to(device)
            model = network_list[k]
            data_num = data.shape[0]
            T = torch.linspace(-0.5, 0.5, data_num, dtype=torch.float32).reshape(-1, 1).to(device)
            T_1 = torch.linspace(-0.75, 0.75, data_num, dtype=torch.float32).reshape(-1, 1).to(device)
            predicts = model(T)
            predicts_1 = model(T_1)
            model.loss = model.loss_function(predicts, data)
            model.loss_list.append(model.loss.cpu().detach().numpy())
            total_predicts_data.append(predicts_1)

        # 节点关系
        if len(data_set) > 1:
            for k in range(len(data_set)):
                model = network_list[k]
                # 计算额外的loss
                if k == 0:
                    loss = torch.sum(((total_predicts_data[k])[-1, :] - (total_predicts_data[k+1])[0, :])**2)
                elif k == len(data_set) - 1:
                    loss = torch.sum(((total_predicts_data[k])[0, :] - (total_predicts_data[k-1])[-1, :])**2)
                else:
                    loss = torch.sum(((total_predicts_data[k])[-1, :] - (total_predicts_data[k+1])[0, :])**2) + torch.sum(((total_predicts_data[k])[0, :] - (total_predicts_data[k-1])[-1, :])**2)
                loss = float(loss.cpu().detach().numpy())
                model.loss += loss * 0.01

        # 反馈
        for k in range(len(data_set)):
            model = network_list[k]
            model.optimizer.zero_grad()
            model.loss.backward()
            model.optimizer.step()

    # # 绘制训练过程图
    # for k in range(len(data_set)):
    #     model = network_list[k]
    #     fig = plt.figure()
    #     t = list(range(len(model.loss_list)))
    #     plt.plot(t, model.loss_list)
    #     plt.xlabel('Epoch Number')
    #     plt.ylabel('Loss Value')
    #     plt.show()
    
    return network_list


## 数据预处理
def data_preprocessing(data, Lb, Ub):

    """
    返回预处理后的data_set
    """

    # 数据去重和归一化
    data = np.unique(data.copy(), axis=0)
    if data.shape[0] == 1:
        return [data]
    data = (data - Lb) / (Ub - Lb)

    # 数据聚类，得到的data_set是一个list，每个元素都是N_k行D列的data数据。k为该类的样本数
    data_set = data_cluster(data)

    return data_set

## 网络训练
def train_net(data_set, epochs, hidden_dim=1000):

    """
    返回训练后的网络
    """

    # 网络训练
    network_list = train([data for data in data_set], epochs, hidden_dim)

    return network_list


# 利用训练好的网络生成预测数据
def generate_data(network_list, data_set, Lb, Ub, total_data_num, NUM):

    """
    NUM为需要生成的数据量
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # 网络使用
    total_predicts_data = []
    for k in range(len(data_set)):
        model = network_list[k]
        data_num = data_set[k].shape[0]
        T = torch.as_tensor(np.random.rand(int(NUM * data_num / total_data_num)).reshape(-1, 1) * 1.5 - 0.75, dtype=torch.float32).to(device)
        predicts_data_list = []
        predicts = model(T)
        count = 0
        for i in range(model.raw_dim):
            if i in model.outputParameter_index_list:
                predicts_data = predicts[:, [count]]
                count += 1
            else:
                predicts_data = torch.ones(T.shape[0], 1) * float(np.mean((data_set[k])[:, [i]]))
            predicts_data_list.append(predicts_data.cpu().detach().numpy())
        predicts_data_list = np.hstack(predicts_data_list)
        total_predicts_data.append(predicts_data_list)
    total_predicts_data = np.vstack(total_predicts_data)
    total_predicts_data = total_predicts_data * (Ub - Lb) + Lb

    return total_predicts_data

## 绘图
def draw(data, total_predicts_data):
    fig = plt.figure()
    if data.shape[1] == 2:
        plt.plot(data[:, 0], data[:, 1], 'o', markersize=3, color='blue', label='current solutions', )  # 绘制真实数据的图
        plt.plot(total_predicts_data[:, 0], total_predicts_data[:, 1], 'o', markersize=3, color='orange', label='generated candidate solutions')  # 绘制生成的数据的图
        plt.xlabel('x1')
        plt.ylabel('x2')
    elif data.shape[1] == 3:
        ax = fig.gca(projection='3d')
        ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o', markersize=3, color='blue', label='current solutions', )  # 绘制真实数据的图
        ax.plot(total_predicts_data[:, 0], total_predicts_data[:, 1], total_predicts_data[:, 2], 'o', markersize=3, color='orange', label='generated candidate solutions')  # 绘制生成的数据的图
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
    plt.legend()
    plt.show()

