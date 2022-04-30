import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from resnet import ResNet18
from clients import ClientsGroup, client
from Norm_ResNet import Norm_ResNet18


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='GTSRB', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.3, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=10, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints2/GTSRB_NIID/', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-lgmr', '--load_global_model_round', type=int, default=-1, help='the round to load trained global model')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def poisoned_client_create(myClients, clients_in_comm, p_num, iid=0, target_class=0):
    poisoned_client = []
    assert p_num < len(clients_in_comm), 'too many poisoned clients'
    for i in range(p_num):
        for c in clients_in_comm:
            if iid == 0:
                if target_class not in myClients.clients_set[c].train_ds.tensors[1] and (c not in poisoned_client):
                    poisoned_client.append(c)
                    break
            else:
                poisoned_client.append(c)
                break
    for p_c in poisoned_client:
        myClients.create_poisoned_data(p_c,iid)
    return poisoned_client

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    test_mkdir(args['save_path'])
    model_save_dir = args['save_path']


    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dataset = datasets.CIFAR("cifar10")

    net = Norm_ResNet18(ResNet18(), 'dataset')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('GTSRB', args['IID'], args['num_of_clients'], dev, 4)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    save_round = [29, 299, 319, 339, 359, 379, 399]
    if args['load_global_model_round'] != -1:
        file_name = os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                args['load_global_model_round'], args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction']))
        global_parameters = torch.load(file_name).state_dict()
        current_round = args['load_global_model_round']
    else:
        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        current_round = 0

    try:
        assert args['num_comm'] > current_round, '联邦学习终止回合应该大于当前回合'
        print('联邦学习终止回合{}'.format(args['num_comm']))
        print('当前回合{}'.format(current_round+1))
    except Exception as ex:
        print('当前回合大于联邦学习终止回合')
    for i in range(current_round+1, args['num_comm']):
        if i < 10:
            args['save_freq'] = 1
        else:
            args['save_freq'] = 10
        print("communicate round {}".format(i+1))
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[:num_in_comm]]
        n_samples = np.array([len(myClients.clients_set[c].train_ds) for c in clients_in_comm])
        sum_parameters = None

        # FL training
        for client in tqdm(clients_in_comm):
            beta = (args['learning_rate']) * (len(myClients.clients_set[client].train_ds)/n_samples.sum())
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = beta * (var.clone() - global_parameters[key].clone())
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + beta * (local_parameters[var] - global_parameters[var])

        for var in global_parameters:
            global_parameters[var] = global_parameters[var] + sum_parameters[var]
        if i in save_round:
            torch.save(global_parameters, os.path.join(model_save_dir,'global_model{}'.format(i+1)))


        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                acc_list = []
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                acc_list.append(sum_accu / num)
                print('accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

