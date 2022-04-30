# Backdoor backdoor detection in FL with similarity measurement

The implementation of federated average learning [1]  based on  PyTorch.



### environment

##### PyTorch-version
1. GPU RTX3090 + torch1.7.1(cu110) + torchvision 0.8.0

### prepare datasets
Before run the code, you need to download train dataset and test data of GTSRB from https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads
and move them to GTSRB dirctory.

### prepare model

You are supposed to prepare the model in ./checkpoints2/,
which is required by load_round in sim_compare1.py. We have given two models for testing.
Run server.py first to obtain the required model at the specific round in load_round.
Run sim_compare1.py to complete detection of backdoor attack. 


### usage

Run the code
python server.py -nc 100 -cf 0.1 -E 5 -B 10 -mn GTSRB  -ncomm 300 -iid 0 -lr 0.01 -vf 1 -g 0
python sim_compare1 -nc 100 -cf 0.1 -E 5 -B 10 -mn GTSRB -ncomm 300 -iid 0 -lr 0.1 -vf 1 -g 0

which means there are 100 clients, we randomly select 10 in each communicating round.  The data set are allocated in Non-IID way.  The epoch and batch size are set to 5 and 10. The learning rate is 0.1, we validate the codes every 1 rounds during the training, training stops after 300 rounds.  



[1] Mcmahan H B , Moore E , Ramage D , et al. Communication-Efficient Learning of Deep Networks from Decentralized Data[J]. 2016.