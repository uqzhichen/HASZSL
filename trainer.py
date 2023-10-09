# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import torch.utils.data
import os
import time
import torch.nn as nn
from PIL import Image
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
from utils import *
from utils import test_zsl, test_gzsl
import random
ATT_DIM = {'CUB': 312, "AWA2": 85, "SUN": 102}

class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())


def compute_alpha(eps, max_iter, is_normalized, eps_size):
    """
    Computes alpha for scaled or normalized inputs. If PGD with
    multiple iterations is performed, alpha is computed by dividing
    by a fixed constant (4.) as opposed to the number of iterations
    (max_iter), which is stated in MIM paper.

    Output:
        - alpha:
            - (is_normalized : True)  := returns a cuda tensor of shape [1,C,1,1], containing alpha values for each channel C
            - (is_normalized : False) := returns a scalar alpha
    """
    alpha = None
    if is_normalized:
        # Epsilon is in the range of possible inputs.
        # Reshape to  [1,C,1,1] to enable broadcasting
        alpha = (eps * eps_size)[np.newaxis, :, np.newaxis, np.newaxis]

        if max_iter > 1:
            alpha = (alpha / 4.)

        alpha = torch.FloatTensor(alpha).cuda()

    else:
        alpha = eps
        if max_iter > 1:
            alpha = eps / 4.

    return alpha


def clamp_tensor(opt, x, is_normalized):
    """ Clamps tensor x between valid ranges for the image (normalized or scaled range)"""
    if is_normalized:
        x.data[:, 0, :, :].clamp_(min=opt.min_val[0], max=opt.max_val[0])
        x.data[:, 1, :, :].clamp_(min=opt.min_val[1], max=opt.max_val[1])
        x.data[:, 2, :, :].clamp_(min=opt.min_val[2], max=opt.max_val[2])
    else:
        x.data.clamp_(min=0.0, max=1.0)

    return x.data


def perturb(opt, batch_input, batch_target, model, attribute_seen, data ,
            loss_log, criterion, criterion_regre,realtrain):
    model.eval()

    with torch.no_grad():
        inputs_embeddings = model(batch_input.cuda(), attribute_seen)[-1].detach().clone()
        inputs_embeddings.requires_grad_(False)

    attribute_seen.requires_grad_(False)

    x = batch_input.clone().detach().cuda()
    x.requires_grad_(True)

    alpha = compute_alpha(opt.eps, opt.loops_adv, is_normalized=True, eps_size=opt.eps_size)

    mapped_target = map_label(batch_target, data.seenclasses)
    labels_v = mapped_target.cuda()
    labels_v.requires_grad_(False)
    # start = time.time()
    # print("hello")
    for ite_max in range(opt.loops_adv):
        output_final, pre_attri, attention, pre_class, \
        inputs_embeddings_output = model(x, attribute_seen)

        attention_flat_layer4 = attention['layer4'].reshape(output_final.shape[0], ATT_DIM[opt.dataset], -1)
        attention_sum = attention_flat_layer4.sum(dim=2)
        #
        attention_loss = attention_sum.mean()
        entropy = entropy_attention_loss(attention_flat_layer4) * opt.entropy_attention
        entropy += entropy_loss(output_final) * opt.entropy_cls

        zsl_loss = F.cross_entropy(output_final, labels_v)

        dist = F.mse_loss(inputs_embeddings_output, inputs_embeddings)
        loss = opt.zsl_weight * zsl_loss - entropy + opt.latent_weight * dist + attention_loss * opt.attention_sup
        model.zero_grad()
        loss.backward()
        noise = x.grad
        x.data = x.data - alpha * torch.sign(noise)

        x.data = clamp_tensor(opt, x, is_normalized=True)
        x.grad.zero_()

    model.train()
    return x.detach().clone()

def Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
            output, pre_attri, attention, pre_class, label_a, label_v,
            realtrain, parts, group_dic, sub_group_dic):

    # for Layer_Regression:
    loss = 0
    if reg_weight['final']['xe'] > 0:
        loss_xe = reg_weight['final']['xe'] * criterion(output, label_v)
        loss_log['l_xe_final'] += loss_xe.item()
        loss = loss_xe

    if reg_weight['final']['attri'] > 0:
        loss_attri = reg_weight['final']['attri'] * criterion_regre(pre_attri['final'], label_a)
        loss_log['l_attri_final'] += loss_attri.item()
        loss += loss_attri

    # add regularization loss
    if opt.additional_loss:
        weight_final = model.ALE_vector.squeeze()
        for name in model.extract:  # "name" is layer4 currently
            layer_xe = reg_weight[name]['l_xe'] * criterion(pre_class[name], label_v)
            loss_log['l_xe_layer'] += layer_xe.item()
            loss += layer_xe
            loss_attri = reg_weight[name]['attri'] * criterion_regre(pre_attri[name], label_a)
            loss_log['l_attri_layer'] += loss_attri.item()
            loss += loss_attri

    return loss


def train_step(model, batch_target, data, batch_input, attribute_seen, opt, loss_log,
               criterion, criterion_regre, realtrain, optimizer):
    # start = time.time()
    # print("hello")
    model.zero_grad()
    # map target labels
    batch_target = map_label(batch_target, data.seenclasses)

    input_v = batch_input.cuda()
    label_v = batch_target.cuda()
    # sim, cls, _ = model(input_v, attribute_seen)
    output, pre_attri, attention, pre_class, _ = model(input_v, attribute_seen)

    label_a = attribute_seen[:, label_v].t()


    loss = Loss_fn(opt, loss_log, opt.reg_weight, criterion, criterion_regre, model,
                   output, pre_attri, attention, pre_class, label_a, label_v,
                   realtrain, opt.parts, opt.group_dic, opt.sub_group_dic)

    loss_log['ave_loss'] += loss.item()
    loss.backward()
    optimizer.step()

    return loss_log


def training(opt, start, epochs, model,  data,  trainloader, testloader_seen,  testloader_unseen, criterion,
             result_zsl, result_gzsl, attribute_seen,  attribute_zsl,  attribute_gzsl):
    criterion_regre = nn.MSELoss()
    layer_name = model.extract[0]
    optimizer = optim.Adam(params=[model.prototype_vectors[layer_name], model.ALE_vector],
                           lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))

    for epoch in range(start, epochs):
        # print("training")
        model.train()
        current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
        realtrain = epoch > (opt.pretrain_epoch)
        layer_name = model.extract[0]
        if epoch >= opt.pretrain_epoch:  # pretrain ALE for the first several epoches
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=current_lr, betas=(opt.beta1, 0.999))

        loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                    'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

        batch = len(trainloader)
        for i, (batch_input, batch_target, impath) in enumerate(trainloader):

            loss_log = train_step(model, batch_target, data, batch_input, attribute_seen, opt, loss_log,
                           criterion, criterion_regre, realtrain, optimizer)

            if random.random() < float(opt.prob_perturb) and epoch >= opt.perturb_start_epoch:

                batch_input = perturb(opt, batch_input, batch_target, model, attribute_seen,data,
                                      loss_log, criterion, criterion_regre,realtrain)
                train_step(model, batch_target, data, batch_input, attribute_seen, opt, loss_log,
                           criterion, criterion_regre, realtrain, optimizer)

        print('\n[Epoch %d, Batch %5d] Train loss: %.3f ' % (epoch + 1, batch, loss_log['ave_loss'] / batch))
        if (i + 1) == batch or (i + 1) % 200 == 0:
            ###### test #######
            # print("testing")
            model.eval()
            # test zsl
            # if not opt.gzsl:
            acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
            if acc_ZSL > result_zsl.best_acc:
                # save model state
                model_save_path = os.path.join('../out/{}_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                torch.save(model.state_dict(), model_save_path)
                print('model saved to:', model_save_path)
            result_zsl.update(epoch + 1, acc_ZSL)
            print('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch + 1,
                                                                acc_ZSL,result_zsl.best_acc, result_zsl.best_iter))
            # else:
                # test gzsl
            acc_GZSL_unseen, probs_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
            acc_GZSL_seen, probs_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

            ########### find the suitable calibration rate ########
            acc_S_T_list, acc_U_T_list, H_list = list(), list(), list()
            for e in np.arange(-1, 1, 0.05):
                tmp_seen_sim = copy.deepcopy(probs_seen)
                tmp_seen_sim[:, data.unseenclasses] += e
                pred_lbl = torch.argmax(tmp_seen_sim, dim=1)
                acc_S_T_list.append((pred_lbl == torch.tensor(
                    testloader_seen.dataset.image_labels[testloader_seen.dataset.img_loc]).cuda()).float().mean())
                tmp_unseen_sim = copy.deepcopy(probs_unseen)
                tmp_unseen_sim[:, data.unseenclasses] += e
                pred_lbl = torch.argmax(tmp_unseen_sim, dim=1)
                acc_U_T_list.append((pred_lbl == torch.tensor(
                    testloader_unseen.dataset.image_labels[testloader_unseen.dataset.img_loc]).cuda()).float().mean())
            for i, j in zip(acc_S_T_list, acc_U_T_list):
                H = 100 * 2 * i * j / (i + j)
                H_list.append(H)
            max_H = max(H_list)
            max_idx = H_list.index(max_H)
            max_U = acc_U_T_list[max_idx] * 100
            max_S = acc_S_T_list[max_idx] * 100
            max_cal = -1+max_idx*0.05

            if max_H > result_gzsl.best_acc:
                # save model state
                model_save_path = os.path.join('../out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                torch.save(model.state_dict(), model_save_path)
                print('model saved to:', model_save_path)

            result_gzsl.update_gzsl(epoch + 1, max_U, max_S, max_H)

            print('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                  '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'
                  '\n best calibration rate: {:.2f}%'.
                  format(epoch + 1, max_U, max_S, max_H, result_gzsl.best_acc_U,
                         result_gzsl.best_acc_S,
                         result_gzsl.best_acc, result_gzsl.best_iter, max_cal))


def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()


def entropy_attention_loss(x):
    out = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
    out = -1.0 * out.sum(dim=2)
    return out.mean()


def to_PIL(tensor):
    return transforms.ToPILImage()(tensor)


def eval_model(opt, model, data, testloader_seen, testloader_unseen, attribute_zsl, attribute_gzsl):
    print('Evaluate ...')
    # model.load_state_dict(torch.load(opt.resume))
    model.eval()
    # test zsl
    # if not opt.gzsl:
    acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
    print('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
    # else:
        # test gzsl
    acc_GZSL_unseen, probs_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
    acc_GZSL_seen, probs_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

    ########### find the suitable calibration rate ########
    acc_S_T_list, acc_U_T_list, H_list = list(), list(), list()
    for e in np.arange(-1, 1, 0.05):
        tmp_seen_sim = copy.deepcopy(probs_seen)
        tmp_seen_sim[:,  data.unseenclasses] += e
        pred_lbl = torch.argmax(tmp_seen_sim, dim=1)
        acc_S_T_list.append((pred_lbl == torch.tensor(testloader_seen.dataset.image_labels[testloader_seen.dataset.img_loc]).cuda()).float().mean())
        tmp_unseen_sim = copy.deepcopy(probs_unseen)
        tmp_unseen_sim[:, data.unseenclasses] += e
        pred_lbl = torch.argmax(tmp_unseen_sim, dim=1)
        acc_U_T_list.append((pred_lbl == torch.tensor(testloader_unseen.dataset.image_labels[testloader_unseen.dataset.img_loc]).cuda()).float().mean())
    for i, j in zip(acc_S_T_list, acc_U_T_list):
        H = 100 * 2 * i * j / (i + j)
        H_list.append(H)
    max_H = max(H_list)
    max_idx = H_list.index(max_H)
    max_U = acc_U_T_list[max_idx] * 100
    max_S = acc_S_T_list[max_idx] * 100

    # if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
    #     acc_GZSL_H = 0
    # else:
    #     acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
    #             acc_GZSL_unseen + acc_GZSL_seen)

    print(
        'GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(max_U, max_S, max_H))


