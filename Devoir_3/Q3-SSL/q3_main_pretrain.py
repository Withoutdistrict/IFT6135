# import requirements
import json
import math
import os
import random
import shutil
import warnings
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy
import numpy as np

from q3_solution import SimSiam

from q3_misc import TwoCropsTransform, load_checkpoints, load_pretrained_checkpoints

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The parameters you will use

# general
seed = 2022
num_workers = 2
save_path = './'
resume = None  # None or a path to a pretrained model (e.g. *.pth.tar')
start_epoch = 0
# epochs = 10  # Number of epoches (for this question 200 is enough, however for 1000 epoches, you will get closer results to the original paper)

# data
dir = './data'
batch_size = 1024

# Siamese backbone model
arch = "resnet18"
fix_pred_lr = True  # fix the learning rate of the predictor network

# Simsiam params
# dim = 2048
pred_dim = 512

# ablation experiments
# stop_gradient = True  # (True or False)
# MLP_mode = None  # None|'no_pred_mlp'

# optimizer
lr = 0.03
momentum = 0.9
weight_decay = 0.0005

# knn params
knn_k = 200  # k in kNN monitor
knn_t = 0.1  # softmax temperature in kNN monitor; could be different with moco-t

# define train and test augmentations for pretraining step
train_transform = [
    transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# datasets and loaders
train_data = datasets.CIFAR10(root=dir, train=True, transform=TwoCropsTransform(transforms.Compose(train_transform)), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

memory_data = datasets.CIFAR10(root=dir, train=True, transform=test_transform, download=True)
memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

test_data = datasets.CIFAR10(root=dir, train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# train for one epoch
def train(train_loader, model, optimizer, device):
    # switch to train mode
    model.train()

    losses = []
    for i, (images, _) in enumerate(train_loader):

        if device is not None:
            images[0] = images[0].to(device, non_blocking=True)
            images[1] = images[1].to(device, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = model.loss(p1, p2, z1, z2, similarity_function='CosineSimilarity')

        losses.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


# save checkpoints
def save_state(model, model_dir, model_name):
    model_dir = os.path.join(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    # with open(os.path.join(model_dir, model_name + ".pickle"), 'wb') as f:
    #     pickle.dump(model, f)
    torch.save(model, os.path.join(model_dir, model_name + ".pth.tar"))


def save_log(dictionary, log_dir, log_name):
    log_dir = os.path.join(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, log_name + ".json"), "w") as f:
        json.dump(dictionary, f, indent=2)


# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, device, knn_k, knn_t):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for i, (data, target) in enumerate(memory_data_loader):
            feature = net(data.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for i, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# adjust LR
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':

    epochs = 200
    dims = [2048, 4096]
    sgs = [True, False]  # (True or False)
    mlp_mode = [None, "no_pred_mlp"]  # None|'no_pred_mlp'
    model_params = [["sgT", sgs[0], mlp_mode[0], dims[0]],
                    ["sgF", sgs[1], mlp_mode[0], dims[0]],

                    ["prP", sgs[0], mlp_mode[0], dims[0]],
                    ["prN", sgs[0], mlp_mode[1], dims[0]],

                    ["dim4096", sgs[0], mlp_mode[0], dims[1]]]

    for model_param in model_params:
        print("\n\n\n\n", model_param)
        model_name, stop_gradient, MLP_mode, dim = model_param

        # set seeds
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

        # Simsiam Model
        print("=> creating model '{}'".format(arch))
        model = SimSiam(models.__dict__[arch], dim, pred_dim, stop_gradient=stop_gradient, MLP_mode=MLP_mode)
        model.to(device)

        # define and set learning rates
        init_lr = lr * batch_size / 256
        if fix_pred_lr:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = model.parameters()
        optimizer = torch.optim.SGD(optim_params, init_lr, momentum=momentum, weight_decay=weight_decay)

        logger = dict()
        logger["t_loss"], logger["t_accu"] = [], []

        # train loop
        for epoch in range(start_epoch, epochs):

            adjust_learning_rate(optimizer, init_lr, epoch, epochs)

            # train for one epoch
            losses = train(train_loader, model, optimizer, device)
            print('Train Epoch: [{}/{}] Train Loss:{:.5f}'.format(epoch, epochs, np.array(losses).mean()))
            logger["t_loss"].append(numpy.array(losses).mean())

            acc1 = test(model.encoder, memory_loader, test_loader, device, knn_k, knn_t)
            print('Test Epoch: [{}/{}] knn_Acc@1: {:.2f}%'.format(epoch, epochs, acc1))
            logger["t_accu"].append(acc1)

            if epoch % 20 == 0:
                state = {'epoch': epoch, 'arch': arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }
                save_state(state, f"models/model_{model_name}", f"model_{epoch}")

        state = {'epoch': epoch, 'arch': arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }
        save_state(state, f"models/model_{model_name}", f"model_{epoch}")
        save_log(logger, f"logs/pretrain", f"log_{model_name}")
