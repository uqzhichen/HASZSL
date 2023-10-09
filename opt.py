import argparse
import glob
import os
import numpy as np
import json


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CUB', help='CUB, AWA2, and SUN')
    parser.add_argument('--root', default='../', help='path to project')
    parser.add_argument('--image_root', default='/home/uqzche20/Datasets/', type=str, metavar='PATH',
                        help='path to image root')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--classifier_lr', type=float, default=1e-6, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=3131, help='manual seed 3483')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--size', type=int, default="448")
    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--pretrained', default=None, help="path to pretrain classifier (to continue training)")
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate to pretrain model')
    parser.add_argument('--all', action='store_true', default=False)

    parser.add_argument('--gzsl', action='store_true')
    parser.add_argument('--additional_loss', action='store_true', default=True)

    parser.add_argument('--xe', type=float, default=1)
    parser.add_argument('--attri', type=float, default=1e-2)
    parser.add_argument('--l_xe', type=float, default=1)
    parser.add_argument('--l_attri', type=float, default=5e-2)

    parser.add_argument('--calibrated_stacking', type=float, default=0.5,
                        help='calibrated_stacking, shrinking the output score of seen classes')

    parser.add_argument('--avg_pool', action='store_true')

    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--resume', default="")

    parser.add_argument('--perturb_lr', default=3, type=float)
    parser.add_argument('--loops_adv', default=30, type=int)
    parser.add_argument('--entropy_cls', default=10, type=float)
    parser.add_argument('--entropy_attention', default=-3, type=float)
    parser.add_argument('--latent_weight', default=1, type=float)
    parser.add_argument('--sim_weight', default=30, type=float)
    parser.add_argument('--zsl_weight', default=1, type=float)
    parser.add_argument('--attention_sup', default=0.1, type=float)
    parser.add_argument('--perturb_start_epoch', default=0, type=int)
    parser.add_argument('--prob_perturb', default=0.5, type=float)
    parser.add_argument('--weight_perturb', default=8.0, type=float)

    opt = parser.parse_args()

    opt.dataroot = opt.root + 'data'

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    opt.max_val = np.array([(1. - mean[0]) / std[0],
                        (1. - mean[1]) / std[1],
                        (1. - mean[2]) / std[2],
                        ])

    opt.min_val = np.array([(0. - mean[0]) / std[0],
                        (0. - mean[1]) / std[1],
                        (0. - mean[2]) / std[2],
                        ])

    opt.eps_size = np.array([abs((1. - mean[0]) / std[0]) + abs((0. - mean[0]) / std[0]),
                         abs((1. - mean[1]) / std[1]) + abs((0. - mean[1]) / std[1]),
                         abs((1. - mean[2]) / std[2]) + abs((0. - mean[2]) / std[2]),
                         ])

    opt.eps = opt.weight_perturb/255.


    # define attribute groups
    if opt.dataset == 'CUB':
        opt.parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
        opt.group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8.json')))
        opt.sub_group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8_layer.json')))
        opt.resnet_path = '../pretrained_models/resnet101_c.pth.tar'
    elif opt.dataset == 'AWA2':
        opt.parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
                 'character']
        opt.group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
        opt.sub_group_dic = {}
        opt.resnet_path = '../pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'SUN':
        opt.parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
        opt.group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
        opt.sub_group_dic = {}
        opt.resnet_path = '../pretrained_models/resnet101_sun.pth.tar'
    else:
        opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'

    opt.reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri},  # l denotes layer
                  }

    print('opt:', opt)

    return opt
