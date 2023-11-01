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

# define train and test augmentation for linear evaluation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_dataset = datasets.CIFAR10(dir, transform=train_transform, download=True, train=True)
val_dataset = datasets.CIFAR10(dir, transform=val_transform, download=True, train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,  # (train_sampler is None),
    num_workers=num_workers, pin_memory=True)  # , sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)


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


# train for one epoch
def train(train_loader, model, criterion, optimizer, device):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    losses = []
    top1 = []
    top5 = []
    for i, (images, target) in enumerate(train_loader):

        if device is not None:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        top1.append(acc1[0].cpu())
        top5.append(acc5[0].cpu())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return top1


# validation
def validate(val_loader, model, criterion, device):
    # switch to evaluate mode
    model.eval()
    losses = []
    top1 = []
    top5 = []
    with torch.no_grad():

        for i, (images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            top1.append(acc1[0].cpu())
            top5.append(acc5[0].cpu())

    return top1


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

        # linear eval
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        # print(model)

        # define and set learning rates
        init_lr = lr * batch_size / 256
        optim_params = model.parameters()
        optimizer = torch.optim.SGD(optim_params, init_lr, momentum=momentum, weight_decay=weight_decay)

        # load the pre-trained model from previous steps
        model_path = f"models/model_{model_name}/model_{199}.pth.tar"
        if model_path:
            model, optimizer, start_epoch = load_pretrained_checkpoints(os.path.join(model_path), model, optimizer, device)
        if device is not None:
            model.to(device)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)

        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        optimizer = torch.optim.SGD(parameters, init_lr, momentum=momentum, weight_decay=weight_decay)

        logger = dict()
        logger["t_accu"], logger["v_accu"] = [], []
        # train for the classififcation task
        for epoch in range(start_epoch, epochs):
            adjust_learning_rate(optimizer, init_lr, epoch, epochs)

            # train for one epoch
            acc1 = train(train_loader, model, criterion, optimizer, device)
            print('Train Epoch: [{}/{}] Train acc1:{:.2f}%'.format(epoch, epochs, np.array(acc1).mean()))
            logger["t_accu"].append(numpy.array(acc1).mean(dtype=float))

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, device)
            print('Val Epoch: [{}/{}] Val acc1:{:.2f}%'.format(epoch, epochs, np.array(acc1).mean()))
            logger["v_accu"].append(numpy.array(acc1).mean(dtype=float))

        save_log(logger, f"logs/classification", f"log_{model_name}")
