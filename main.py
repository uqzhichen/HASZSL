from __future__ import print_function
import torch.backends.cudnn as cudnn
from model import resnet_proto_IoU
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
from utils import test_zsl, calibrated_stacking, test_gzsl, set_randomseed, Result
from dataset import get_loader, DATA_LOADER
from opt import get_opt
from trainer import *
import glob

cudnn.benchmark = True

opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device


def main():
    # load data
    data = DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label

    # prepare the attribute labels
    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()

    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen = get_loader(opt, data)

    # initialize model
    print('Create Model...')
    model = resnet_proto_IoU(opt)
    criterion = nn.CrossEntropyLoss()


    opt.reg_lambdas = {}
    for name in ['final'] + model.extract:
        opt.reg_lambdas[name] = opt.reg_weight[name]
    # print('reg_lambdas:', reg_lambdas)

    if torch.cuda.is_available():
        model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_gzsl = attribute_gzsl.cuda()

    # train and test
    result_zsl = Result()
    result_gzsl = Result()

    if opt.only_evaluate:
        state = torch.load(opt.resume)
        model.load_state_dict(state)
        eval_model(opt, model, data, testloader_seen, testloader_unseen, attribute_zsl, attribute_gzsl)
    else:
        training(opt, 0, 60, model, data, trainloader, testloader_seen, testloader_unseen,
                 criterion, result_zsl, result_gzsl, attribute_seen, attribute_zsl, attribute_gzsl)

if __name__ == '__main__':
    main()

