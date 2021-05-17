import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def data_cluster(data):
    """
    数据聚类。
    方法：先用DBScan，如果发现聚类数目过多，则采用k-means
    """

    distance_matrix = distance.cdist(data, data, 'euclidean')

    y_pred = DBSCAN(eps = np.mean(distance_matrix)).fit_predict(data)
    cluster_num = len(np.unique(y_pred))
    if cluster_num > data.shape[0] // 4:
        y_pred = KMeans(n_clusters=2).fit_predict(data)
    
    # y_pred = KMeans(n_clusters=2).fit_predict(data)

    unique_cluster_num = np.unique(y_pred)

    data_set = []
    for i in range(len(unique_cluster_num)):
        data_set.append(data[np.where(y_pred == unique_cluster_num[i])[0], :])
    
    return data_set

## 数据排序
def sort_data(data):
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
    return data[order, :]

## 子网络
class SubNet(nn.Module):
    def __init__(self, hidden_dim=1000, epochs=1000):
        """
        子网络：拟合隐变量与某一维数据的关系
        """
        super(SubNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.loss_list = []  # 用于记录训练过程的loss value
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.epochs = epochs
        self.trained = False

    def forward(self, x):
        x = self.model(x)
        return x

def train(data_set, epochs=1000):
    """
    注意：传入的data_set是聚类后的数据,已经归一化,无需再归一化.这里只需对data_set的每一类数据再进行排序即可
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # 存储所有的网络输出数据
    total_predicts_data = []

    # 创建网络
    network_list = []
    
    for k in range(len(data_set)):
        data = data_set[k].copy()

        # 数据排序
        if data.shape[0] > 1:
            data = sort_data(data)
        
        data = torch.as_tensor(data, dtype=torch.float32).to(device)
        Dim = data.shape[1]
        data_num = data.shape[0]
        T = torch.linspace(-0.5, 0.5, data_num, dtype=torch.float32).reshape(-1, 1).to(device)  # 创建隐变量
        model_list = []
        for i in range(Dim):
            model_list.append(SubNet(epochs=epochs).to(device))
        
        # 训练网络
        for i in range(Dim):
            # print('model: ', i)
            model = model_list[i]
            sub_data = data[:, [i]]
            # 如果sub_data的标准差过小，则不需要训练
            if torch.std(sub_data) > 1e-2:
                for epoch in range(1, model.epochs+1):
                    # print(epoch)
                    predicts = model(T)
                    loss = model.loss_function(predicts, sub_data)
                    model.loss_list.append(loss.cpu().detach().numpy())
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.optimizer.step()
                model.trained = True  # 标记已经训练过
            # print(np.max(model.loss_list))
            

        # # 绘制训练过程图
        # fig = plt.figure()
        # for i in range(Dim):
        #     t = list(range(len(model_list[i].loss_list)))
        #     plt.plot(t, model_list[i].loss_list, label=f'model_{str(i)}')
        #     plt.legend()
        #     plt.xlabel('Epoch Number')
        #     plt.ylabel('Loss Value')
        # plt.show()
        network_list.append(model_list)
    
    return network_list

def generate_data(data, Lb, Ub, NUM, epochs):

    """
    NUM为需要生成的数据量
    """
    ## Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # 数据去重和归一化
    data = np.unique(data.copy(), axis=0)
    if data.shape[0] == 1:
        return data
    data = (data - Lb) / (Ub - Lb)
    total_data_num = data.shape[0]

    # 数据聚类，得到的data_set是一个list，每个元素都是N_k行D列的data数据。k为该类的样本数
    data_set = data_cluster(data)

    # 网络训练
    network_list = train(data_set, epochs)

    # 网络使用
    total_predicts_data = []
    for k in range(len(data_set)):
        model_list = network_list[k]
        data_num = data_set[k].shape[0]
        T = torch.as_tensor(np.random.rand(int(NUM * data_num / total_data_num)).reshape(-1, 1) * 2 - 1, dtype=torch.float32).to(device)
        predicts_data = []
        for i in range(len(model_list)):
            model = model_list[i]
            if model.trained:
                predicts = model(T)
            else:
                predicts = torch.ones(T.shape[0], 1) * float(np.mean((data_set[k])[:, [i]]))
            predicts_data.append(predicts)
        predicts_data = torch.hstack(predicts_data).cpu().detach().numpy()
        total_predicts_data.append(predicts_data)
    total_predicts_data = np.vstack(total_predicts_data)
    total_predicts_data = total_predicts_data * (Ub - Lb) + Lb

    return total_predicts_data