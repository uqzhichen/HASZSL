from tqdm import tqdm
import torch
import numpy as np
from statistics import mean
import torchvision.transforms as transforms
import random
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import torch.utils.data
import os
import numpy as np
import h5py
import torch
import torch.utils.data
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.acc_list = []
        self.epoch_list = []
    def update(self, it, acc):
        self.acc_list += [acc]
        self.epoch_list += [it]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.epoch_list += [it]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s


def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def test_zsl(opt, model, testloader, attribute, test_classes):
    layer_name = model.extract[0]
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    with torch.no_grad():
        for i, (input, target, impath) in \
                enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, _, _, pre_class, _  = model(input, attribute)
            _, predicted_label = torch.max(output.data, 1)
            _, predicted_layer = torch.max(pre_class[layer_name].data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc(map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc(map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                             np.array(predicted_layers), test_classes.numpy())
    if opt.all:
        return acc_all * 100
    else:
        return acc_avg * 100


def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)


def test_gzsl(opt, model, testloader, attribute, test_classes):
    layer_name = model.extract[0]
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    probs = []
    with torch.no_grad():
        for i, (input, target, impath) in \
                enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, pre_attri, _, pre_class, _ = model(input, attribute)
            probs.append(output)
            if opt.calibrated_stacking:
                output = calibrated_stacking(opt, output, opt.calibrated_stacking)
            # small = torch.where((pre_attri['final'] - attribute[:, target].T).abs().sum(dim=1) < 200)
            # if small[0].shape[0] > 0:
            #     print(np.array(impath)[small[0].tolist()])
            _, predicted_label = torch.max(output.data, 1)
            _, predicted_layer = torch.max(pre_class[layer_name].data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    probs = torch.cat(probs, dim=0)
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc_gzsl(GT_targets,
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc_gzsl(GT_targets,
                                             np.array(predicted_layers), test_classes.numpy())
    if opt.all:
        return acc_all * 100, probs
    else:
        return acc_avg * 100, probs


def calculate_average_IoU(whole_IoU, IoU_thr=0.5):
    img_num = len(whole_IoU)
    body_parts = whole_IoU[0].keys()
    body_avg_IoU = {}
    for body_part in body_parts:
        body_avg_IoU[body_part] = []
        body_IoU = []
        for im_id in range(img_num):
            if len(whole_IoU[im_id][body_part]) > 0:
                if_one = []
                for item in whole_IoU[im_id][body_part]:
                    if_one.append(1 if item > IoU_thr else 0)
                body_IoU.append(mean(if_one))
        body_avg_IoU[body_part].append(mean(body_IoU))
    num = 0
    sum = 0
    for part in body_avg_IoU:
        if part != 'tail':
            sum += body_avg_IoU[part][0]
            num += 1
    # print(sum/num *100)
    return body_avg_IoU, sum/num *100


def set_randomseed(opt):
    # define random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    # improve the efficiency
    # check CUDA
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")









def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range(len(nclass)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class) / len(acc_per_class)


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = []

    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class) / len(acc_per_class)


def prepare_attri_label(attribute, classes):
    # print("attribute.shape", attribute.shape)
    classes_dim = classes.size(0)
    attri_dim = attribute.shape[1]
    output_attribute = torch.FloatTensor(classes_dim, attri_dim)
    for i in range(classes_dim):
        output_attribute[i] = attribute[classes[i]]
    return torch.transpose(output_attribute, 1, 0)

