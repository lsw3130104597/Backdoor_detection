import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from torchvision import transforms

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                if label.dtype != torch.float64:
                    label = torch.tensor(label, dtype=torch.long).to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()
    def localUpdate_backdoor_normal(self, is_poi, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, poi_test_loader):
        if not is_poi:
            Net.load_state_dict(global_parameters, strict=True)
            self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
            for epoch in range(localEpoch):
                for data, label in self.train_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    # 将 label类型有int32转换为int64，适用于lossfun函数的参数
                    if label.dtype != torch.int64:
                        label = torch.tensor(label, dtype = torch.long).to(self.dev)
                    preds = Net(data)
                    loss = lossFun(preds, label)
                    loss.backward()
                    opti.step()
                    opti.zero_grad()
                # with torch.no_grad():
                #     if epoch % 5 == 0:
                #         # parameters = torch.load('./model_save/posion_model_retrain2')
                #         # Net.load_state_dict(parameters, strict=True)
                #         sum_accu = 0
                #         num = 0
                #         for data, label in poi_test_loader:
                #             data, label = data.to(self.dev), label.to(self.dev)
                #             preds = Net(data)
                #             preds = torch.argmax(preds, dim=1)
                #             sum_accu += (preds == label).float().mean()
                #             num += 1
                #         print('Backdoor accuracy: {}'.format(sum_accu / num))
        else:
            Net.load_state_dict(global_parameters, strict=True)
            self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
            for epoch in range(localEpoch):
                for data, label in self.train_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = Net(data)
                    # preds = torch.argmax(preds,axis=1) #新加的
                    if label.dtype != torch.int64:
                        label = torch.tensor(label, dtype = torch.long).to(self.dev)
                    loss = lossFun(preds, label)
                    loss.backward()
                    opti.step()
                    opti.zero_grad()
                    
                with torch.no_grad():
                    if epoch % 5 == 0:
                        # parameters = torch.load('./model_save/posion_model_retrain2')
                        # Net.load_state_dict(parameters, strict=True)
                        sum_accu = 0
                        num = 0
                        for data, label in poi_test_loader:
                            data, label = data.to(self.dev), label.to(self.dev)
                            preds = Net(data)
                            preds = torch.argmax(preds, dim=1)
                            sum_accu += (preds == label).float().mean()
                            num += 1
                        print('Backdoor accuracy: {}'.format(sum_accu / num))
                        # if (sum_accu / num)>0.75:
                        #     return  Net.state_dict()
        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, datadim):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation(datadim)

    def dataSetBalanceAllocation(self, datadim):

        DataSet = GetDataSet(self.data_set_name, self.is_iid, datadim)

        test_data = torch.tensor(DataSet.test_data)
        test_label = torch.tensor(DataSet.test_label)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = DataSet.train_data
        train_label = DataSet.train_label

        shard_size = DataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(DataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            #for mnist
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            #for cifar10
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.hstack((label_shards1, label_shards2))

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

    def trigger_tri_r_cifar(self):
        m = np.zeros([32, 32],dtype=int)
        m[27, 27] = 1
        m[27, 28] = 1
        m[28, 28] = 1
        delta = np.array([1.0,0.0,0.0])  # trigger color
        return torch.tensor(m), torch.tensor(delta)
    def trigger_star_r_cifar(self):
        m = np.zeros([32,32],dtype=int)
        m[26, 26] = 1
        m[26, 28] = 1
        m[27, 27] = 1
        m[28, 26] = 1
        m[28, 28] = 1
        delta = [1.0,0.0,0.0]  # trigger color
        return torch.tensor(m), torch.tensor(delta)
    def trigger_point_r_cifar(self):
        m = np.zeros([32,32],dtype=int)
        m[27, 27] = 1
        delta = [1.0,0.0,0.0]  # trigger color
        return torch.tensor(m), torch.tensor(delta)
    def trigger_tri_g_cifar(self):
        m = np.zeros([32,32],dtype=int)
        m[4, 27] = 1
        m[4, 28] = 1
        m[5, 28] = 1
        delta = [0.0,1.0,0.0]  # trigger color
        return torch.tensor(m), torch.tensor(delta)
    def trigger_star_b_cifar(self):
        m = np.zeros([32,32],dtype=int)
        m[4, 4] = 1
        m[4, 6] = 1
        m[5, 5] = 1
        m[6, 4] = 1
        m[6, 6] = 1
        delta = [0.0,0.0,1.0]  # trigger color
        return torch.tensor(m), torch.tensor(delta)

    def get_axis(self, axarr, H, W, i, j):
        H, W = H - 1, W - 1
        if not (H or W):
            ax = axarr
        elif not (H and W):
            ax = axarr[max(i, j)]
        else:
            ax = axarr[i][j]
        return ax

    def create_poisoned_data(self, p_c, iid):
        c_train_data = self.clients_set[p_c].train_ds.tensors[0]
        c_train_label = self.clients_set[p_c].train_ds.tensors[1]
        poison_rate = 1/10
        poison_data_num = int(np.floor(len(c_train_label)*poison_rate))
        np.random.seed(12)
        if iid == 0:
            poison_data_index = np.random.choice(len(c_train_label),poison_data_num,replace=False)
        else:
            non_zero_label = list(torch.where(self.clients_set[p_c].train_ds.tensors[1]!=0))
            non_zero_label = np.array(non_zero_label[0])
            poison_data_index = np.random.choice(non_zero_label,poison_data_num,replace=False)
        triggers = []
        for i in poison_data_index:
            # trigger = np.random.choice([0, 1, 2, 3, 4])
            trigger = np.random.choice([0, 1, 2])
            triggers.append(trigger)
            if trigger == 0:
                t_m, t_delta = self.trigger_point_r_cifar()
            elif trigger == 1:
                t_m, t_delta = self.trigger_tri_r_cifar()
            elif trigger == 2:
                t_m, t_delta = self.trigger_star_r_cifar()
            elif trigger == 3:
                t_m, t_delta = self.trigger_tri_g_cifar()
            else:
                t_m, t_delta = self.trigger_star_b_cifar()
            assert len(c_train_data[i]) == len(t_delta)
            for j in range(len(c_train_data[i])):
                c_train_data[i][j] = (1-t_m) * c_train_data[i][j] + t_delta[j] * t_m
            c_train_label[i] = 0
        for i in range(len(c_train_data)):
            self.clients_set[p_c].train_ds.tensors[0][i] = c_train_data[i]
            self.clients_set[p_c].train_ds.tensors[1][i] = c_train_label[i]


    def create_poisoned_data_cifar(self, p_c, iid):
        c_train_data = self.clients_set[p_c].train_ds.tensors[0].clone()
        c_train_label = self.clients_set[p_c].train_ds.tensors[1].clone()
        poison_rate = 8/10
        poison_data_num = int(np.floor(len(c_train_label)*poison_rate))
        np.random.seed(12)
        if iid == 0:
            poison_data_index = np.random.choice(len(c_train_label),poison_data_num,replace=False)
        else:
            non_zero_label = list(torch.where(self.clients_set[p_c].train_ds.tensors[1]!=0))
            non_zero_label = np.array(non_zero_label[0])
            poison_data_index = np.random.choice(non_zero_label,poison_data_num,replace=False)
        triggers = []
        for i in poison_data_index:
            # trigger = np.random.choice([0, 1, 2, 3, 4])
            trigger = np.random.choice([0,1,2])
            triggers.append(trigger)

            if trigger == 0:
                t_m, t_delta = self.trigger_point_r_cifar()
            elif trigger == 1:
                t_m, t_delta = self.trigger_tri_r_cifar()
            elif trigger == 2:
                t_m, t_delta = self.trigger_star_r_cifar()
            elif trigger == 3:
                t_m, t_delta = self.trigger_tri_g_cifar()
            else:
                t_m, t_delta = self.trigger_star_b_cifar()
            assert len(c_train_data[i]) == len(t_delta)
            for j in range(len(c_train_data[i])):
                c_train_data[i][j] = (1-t_m) * c_train_data[i][j] + t_delta[j] * t_m
            c_train_label[i] = 0



        # a = []
        # for i in range(5):
        #     f = np.where(np.array(triggers) == i)[0][0]
        #     a.append(f)
        # a = np.array(a)
        # b = poison_data_index[a]
        # data = c_train_data[b].clone()
        # fig, axarr = plt.subplots(1, 5, figsize=(2.5 * 5, 2.5), dpi=600)
        # for i in range(5):
        #     ax = axarr[i]
        #     unloader = transforms.ToPILImage()
        #     image = data[i].cpu().clone()  # clone the tensor
        #     image = image.squeeze(0)  # remove the fake batch dimension
        #     image = unloader(image)
        #     ax.imshow(image)
        #     ax.xaxis.set_ticks([])
        #     ax.yaxis.set_ticks([])
        # plt.show()
        # # fig.savefig('poison_image.tiff')
        # # print('aaa')
        return c_train_data, c_train_label, poison_data_index
# data = c_train_data[385]
# from torchvision import transforms
# import matplotlib.pyplot as plt
# unloader = transforms.ToPILImage()
# image = data.cpu().clone()  # clone the tensor
# image = image.squeeze(0)  # remove the fake batch dimension
# image = unloader(image)
# plt.imshow(image)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# from torchvision import transforms
# import matplotlib.pyplot as plt
# plt.figure(1,dpi = 600)
# ax = []
# ax1 = plt.subplot(1, 5, 1)
# ax2 = plt.subplot(1, 5, 2)
# ax3 = plt.subplot(1, 5, 3)
# ax4 = plt.subplot(1, 5, 4)
# ax5 = plt.subplot(1, 5, 5)
# ax.append(ax1)
# ax.append(ax2)
# ax.append(ax3)
# ax.append(ax4)
# ax.append(ax5)
# a = []
# for i in range(5):
#    f = np.where(np.array(triggers) == i )[0][0]
#    a.append(f)
# a = np.array(a)
# b = poison_data_index[a]
# data = c_train_data[b]
# unloader = transforms.ToPILImage()
# for i in range(5):
#     image = data[i].cpu().clone()  # clone the tensor
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#     plt.sca(ax[i])
#     plt.imshow(image)
# plt.show()





if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


