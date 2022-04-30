#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/8/30 19:17
# @Author  : Pipifei
# @Email   : lsw5636@163.com
# @File    : exp1.py
# @Software: PyCharm
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
# from ..Models import Mnist_2NN, Mnist_CNN,GTSRB_CNN, GTSRB_CNN1
from resnet import ResNet18
from clients import ClientsGroup, client
from Norm_ResNet import Norm_ResNet18
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
# from DFTND.robustness_lib.robustness import  datasets

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=10, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints2/GTSRB_NIID/sim_compare_res', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-lgmr', '--load_global_model_round', type=int, default=-1, help='the round to load trained global model')

# Create file_dir
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#Type of backdoor triggers
def trigger_tri_r_cifar():
    m = np.zeros([32, 32],dtype=int)
    m[27, 27] = 1
    m[27, 28] = 1
    m[28, 28] = 1
    delta = np.array([1.0,0.0,0.0])  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_star_r_cifar():
    m = np.zeros([32,32],dtype=int)
    m[26, 26] = 1
    m[26, 28] = 1
    m[27, 27] = 1
    m[28, 26] = 1
    m[28, 28] = 1
    delta = [1.0,0.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_point_r_cifar():
    m = np.zeros([32,32],dtype=int)
    m[27, 27] = 1
    delta = [1.0,0.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_tri_g_cifar():
    m = np.zeros([32,32],dtype=int)
    m[4, 27] = 1
    m[4, 28] = 1
    m[5, 28] = 1
    delta = [0.0,1.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_star_b_cifar():
    m = np.zeros([32,32],dtype=int)
    m[4, 4] = 1
    m[4, 6] = 1
    m[5, 5] = 1
    m[6, 4] = 1
    m[6, 6] = 1
    delta = [0.0,0.0,1.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)

#create poisoned  dataset, where some sample are stamped with trigger and their labels are modified
def create_poison_data (Train_data,Train_label,num):

    np.random.seed(12)
    # if iid == 0:
    #     poison_data_index = np.random.choice(len(c_train_label), poison_data_num, replace=False)
    # else:
    num = int(num)
    non_zero_label = list(torch.where(Train_label != 0))
    non_zero_label = np.array(non_zero_label[0])
    poison_data_index = np.random.choice(non_zero_label, num, replace=False)
    poison_data1 = Train_data[poison_data_index]
    poison_label1 = Train_label[poison_data_index]
    triggers = []
    for i in range(num):
        # trigger = np.random.choice([0, 1, 2, 3, 4])
        trigger = np.random.choice([0, 1, 2])
        triggers.append(trigger)

        if trigger == 0:
            t_m, t_delta = trigger_point_r_cifar()
        elif trigger == 1:
            t_m, t_delta = trigger_tri_r_cifar()
        elif trigger == 2:
            t_m, t_delta = trigger_star_r_cifar()
        elif trigger == 3:
            t_m, t_delta = trigger_tri_g_cifar()
        else:
            t_m, t_delta = trigger_star_b_cifar()
        assert len(Train_data[i]) == len(t_delta)
        for j in range(len(Train_data[i])):
            poison_data1[i][j] = (1 - t_m) * poison_data1[i][j] + t_delta[j] * t_m
        poison_label1[i] = 0
    return poison_data1, poison_label1, poison_data_index

def poisoned_data_create(Train_data, Train_label, sample_num):
    poison_rate = 3/10  # poison sample rates
    poison_num = (poison_rate * sample_num) / (1- poison_rate)
    poison_data, poison_label, poison_index = create_poison_data (Train_data, Train_label, poison_num)
    return  poison_data, poison_label, poison_index

# create poisoned testing dataset
def poisoned_test_data(test_data1, test_label1):
    test_data = test_data1.clone()
    test_label = test_label1.clone()
    assert len(test_data) == len(test_label)
    for i in range(len(test_data)):
        # trigger = np.random.choice([0, 1, 2, 3, 4])
        trigger = np.random.choice([0, 1, 2])
        if trigger == 0:
            t_m, t_delta = trigger_point_r_cifar()
        elif trigger == 1:
            t_m, t_delta = trigger_tri_r_cifar()
        elif trigger == 2:
            t_m, t_delta = trigger_star_r_cifar()
        elif trigger == 3:
            t_m, t_delta = trigger_tri_g_cifar()
        else:
            t_m, t_delta = trigger_star_b_cifar()
        assert len(test_data[i]) == len(t_delta)
        for j in range(len(test_data[i])):
            test_data[i][j] = (1 - t_m) * test_data[i][j] + t_delta[j] * t_m
        test_label[i] = 0
    return test_data, test_label

# PASM(ours)
def  cos_sim1(cur_global_parameters, pre_global_parameters, local_updates):
    update_G = {}
    for key in cur_global_parameters:
        update_G[key] = cur_global_parameters[key].clone() - pre_global_parameters[key].clone()
    update_G_flat = [var.flatten() for key, var in update_G.items()]
    update_G_flat = torch.cat(update_G_flat)
    update_flats = []
    for update in local_updates:
        update_f = [var.flatten() for key, var in update.items()]
        update_f = torch.cat(update_f)
        update_flats.append(update_f)
    threshold = 0.95
    scores = []
    for i in range(len(update_flats)):
        vector_a = np.array(update_G_flat.view(-1,1).cpu())
        vector_b = np.array(update_flats[i].view(-1,1).cpu())
        a = vector_a[vector_a < 1.0]
        b = vector_b[vector_a < 1.0]
        num = np.dot(a,b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = num / denom
        sim = cos
        scores.append(sim)
    print('Abnormal Score of PASM: ', scores)
    poison_client_index =[]
    for i in range(len(scores)):
        if scores[i]>threshold:
            poison_client_index.append(i)
    if len(poison_client_index) == 0:
        return [-1]
    else:
        return poison_client_index
    # print(scores)
# Similarity measurement between two different local model
def  cos_sim2(local_updates):

    update_flats = []
    for update in local_updates:
        update_f = [var.flatten() for key, var in update.items()]
        update_f = torch.cat(update_f)
        update_flats.append(update_f)
    threshold = 0.27
    cos_sim = []
    for i in range(len(update_flats)):
        list1 = []
        for j in range(len(update_flats)):
            if i != j:
                vector_a = np.array(update_flats[i].view(-1, 1).cpu())
                vector_b = np.array(update_flats[j].view(-1, 1).cpu())
                a = vector_a[vector_a < 1.0]
                b = vector_b[vector_a < 1.0]
                num = np.dot(a, b)
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                cos = num / denom
                list1.append(cos)
        cos_sim.append(np.max(list1))
    poison_client_index = []
    poison_client_index.append(np.argmax(cos_sim))
    if len(poison_client_index) == 0:
        return [-1]
    else:
        return poison_client_index

# Similarity measurement between global model and local model
def cos_sim3(pre_global_model, local_updates):
    # pre_global_model = local_model_sets[0]

    pre_global_model_f = [var.flatten() for key, var in pre_global_model.items()]
    pre_global_model_f = torch.cat(pre_global_model_f)
    scores = []
    for model in local_updates:
        model_f = [var.flatten() for key, var in model.items()]
        model_f = torch.cat(model_f)
        vector_a = np.array(model_f.view(-1, 1).cpu())
        vector_b = np.array(pre_global_model_f.view(-1, 1).cpu())
        a = vector_a[vector_a < 1.0]
        b = vector_b[vector_a < 1.0]
        num = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = num / denom
        scores.append(cos)
    # print('Score Detection', scores)
    poison_client_index =  [np.argmax(np.abs(scores))]
    if len(poison_client_index) == 0:
        return [-1]
    else:
        return poison_client_index
    # print(cos_sim)

# detect backdoor model
def cos_sim_detect(local_parameters_set, posion_client_index, gama):
    local_updates = []
    pre_global_parameters = local_parameters_set[0]
    cur_global_parameters = local_parameters_set[-1]
    is_ture1 = 0
    is_ture2 = 0
    is_ture3 = 0
    for i in range(10):
        if i == posion_client_index:
            update = {}
            for key, var in local_parameters_set[i+1].items():
                update[key] = gama * (var.clone() - pre_global_parameters[key].clone())
            local_updates.append(update)
        else:
            update = {}
            for key, var in local_parameters_set[i+1].items():
                update[key] = var.clone() - pre_global_parameters[key].clone()
            local_updates.append(update)
    posion_client_index3 = cos_sim3(pre_global_parameters, local_updates)
    posion_client_index1 = cos_sim1(cur_global_parameters, pre_global_parameters, local_updates)
    posion_client_index2 = cos_sim2(local_updates)

    if posion_client_index in posion_client_index1:
        is_ture1 = 1
    if posion_client_index in posion_client_index2:
        is_ture2 = 1
    if posion_client_index in posion_client_index3:
        is_ture3 = 1

    return is_ture1, is_ture2, is_ture3

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # net = GTSRB_CNN1()
    #If we use Resnet18 to classify GTSRB dataset, modify its out classes to 43;  class 10 for cifar10
    net = Norm_ResNet18(ResNet18(), 'dataset')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('GTSRB', args['IID'], args['num_of_clients'], dev, 4)
    testDataLoader = myClients.test_data_loader

    Train_data = [myClients.clients_set[c].train_ds.tensors[0] for c in myClients.clients_set]
    Train_label = [myClients.clients_set[c].train_ds.tensors[1] for c in myClients.clients_set]
    # sample_num = [len(myClients.clients_set[c].train_ds.tensors[0]) for c in myClients.clients_set]
    Train_data = torch.stack(Train_data).view(-1,3,32,32)
    Train_label = torch.stack(Train_label).view(-1,)
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    # The communication round initiating backdoor attack
    # load_round = [29, 299, 319, 339, 359, 379, 399]
    load_round = [29, 299]
    detect_acc1 = []
    detect_acc2 = []
    detect_acc3 = []

    for i in load_round:
        print('The {}-th round of FL training'.format(i+1))
        file_name = 'checkpoints2/GTSRB_NIID/GTSRB_num_comm{}_E5_B10_lr0.1_num_clients100_cf0.1'.format(i)

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[:num_in_comm]]
        n_samples = np.array([len(myClients.clients_set[c].train_ds) for c in clients_in_comm])

        c_poi_train_data, c_poi_train_label, c_poi_train_index = poisoned_data_create(Train_data, Train_label, n_samples[0])
        test_data_temp, test_label_temp = testDataLoader.dataset.tensors[0].clone(), testDataLoader.dataset.tensors[1].clone()
        c_poi_test_data, c_poi_test_label = poisoned_test_data(test_data_temp[:1000], test_label_temp[:1000])
        poi_test_loader = DataLoader(TensorDataset(c_poi_test_data.clone().detach(), c_poi_test_label.clone().detach()),
                                     batch_size=10, shuffle=False)
        acc1 = 0
        acc2 = 0
        acc3 = 0
        for j in range(num_in_comm):
            sum_parameters = None
            global_parameters = torch.load(file_name).state_dict()
            poison_client =  clients_in_comm[j]
            print ('{}-th backdoor attacker in {}-th round'.format(j,i))

            normal_train_data = myClients.clients_set[poison_client].train_ds.tensors[0].clone()
            normal_train_label = myClients.clients_set[poison_client].train_ds.tensors[1].clone()
            c_poi_train_data = torch.cat((copy.deepcopy(c_poi_train_data),normal_train_data),0)
            c_poi_train_label = torch.cat((copy.deepcopy(c_poi_train_label),normal_train_label),0)

            local_parameters_set = []
            local_parameters_set.append(copy.deepcopy(global_parameters))

            for client in tqdm(clients_in_comm):

                if client == poison_client:
                    is_poi = 1
                    ori_trian_ds = TensorDataset(
                        torch.tensor(myClients.clients_set[client].train_ds.tensors[0].clone().detach()),
                        torch.tensor(myClients.clients_set[client].train_ds.tensors[1].clone().detach()))
                    poi_trian_ds = TensorDataset(torch.tensor(c_poi_train_data.clone().detach()),
                                                 torch.tensor(c_poi_train_label.clone().detach()))
                    myClients.clients_set[client].train_ds = poi_trian_ds
                    local_parameters = myClients.clients_set[client].localUpdate_backdoor_normal(is_poi,30,
                                                                                                 args['batchsize'],
                                                                                                 net,
                                                                                                 loss_func, opti,
                                                                                                 global_parameters,
                                                                                                 poi_test_loader)
                    myClients.clients_set[client].train_ds = ori_trian_ds

                else:
                    is_poi = 0
                    local_parameters = myClients.clients_set[client].localUpdate_backdoor_normal(is_poi, args['epoch'],
                                                                                                 args['batchsize'],
                                                                                                 net,
                                                                                                 loss_func, opti,
                                                                                                 global_parameters, [])
                local_parameters_set.append(copy.deepcopy(local_parameters))
                beta = (args['learning_rate']) * (len(myClients.clients_set[client].train_ds) / n_samples.sum())
                gama1 = 1 / beta

                # 将所有局部模型聚合为全局模型
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        if client == poison_client:
                            sum_parameters[key] = beta * gama1 * (var.clone().detach()
                                                                  - global_parameters[key].clone().detach())
                        else:
                            sum_parameters[key] = beta * (
                                        var.clone().detach() - global_parameters[key].clone().detach())
                else:
                    for var in sum_parameters:
                        if client == poison_client:
                            sum_parameters[var] = sum_parameters[var].clone().detach() + beta * gama1 * (
                                        local_parameters[var].clone().detach() - global_parameters[var].clone().detach())
                        else:
                            sum_parameters[var] = sum_parameters[var].clone().detach() + beta * (
                                        local_parameters[var].clone().detach() -
                                        global_parameters[var].clone().detach())
            for var in global_parameters:
                global_parameters[var] = global_parameters[var].clone().detach() + sum_parameters[
                    var].clone().detach()
            local_parameters_set.append(copy.deepcopy(global_parameters))

            #用三种相似度对模型进行检测
            is_ture1, is_ture2, is_ture3 = cos_sim_detect(local_parameters_set, j, gama1)
            acc1 += is_ture1
            acc2 += is_ture2
            acc3 += is_ture3

        acc1 = acc1/num_in_comm
        acc2 = acc2/num_in_comm
        acc3 = acc3/num_in_comm
        detect_acc1.append(acc1)
        detect_acc2.append(acc2)
        detect_acc3.append(acc3)
    print(detect_acc1)
    print(detect_acc2)
    print(detect_acc3)