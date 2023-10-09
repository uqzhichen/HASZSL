import os
import sys
import copy
import h5py
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
import torch.utils.data
from utils import map_label
from sklearn import preprocessing
import torchvision.transforms as transforms


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        self.image_files = matcontent['image_files'].squeeze()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # print("matcontent.keys:", matcontent.keys())
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        if opt.dataset == 'CUB':
            self.train_loc = matcontent['train_loc'].squeeze() - 1
            self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            # self.train_unseen_loc = matcontent['train_unseen_loc'].squeeze() - 1

        # self.train_loc = matcontent['train_loc'].squeeze() - 1
        # self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.allclasses_name = matcontent['allclasses_names']
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attri_name = matcontent['attri_name']

        # self.train_unseen_loc = matcontent['train_unseen_loc'].squeeze() - 1
        # self.test_unseen_small_loc = matcontent['test_unseen_small_loc'].squeeze() - 1
        # print("allclasses_names", self.allclasses_name)
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[self.trainval_loc])
                _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[self.trainval_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[self.test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[self.test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
            self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[self.val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # print("self.test_unseen_label:", list(set(self.test_unseen_label.numpy())))
        # print("self.unseenclasses:", list(set(self.unseenclasses.numpy())))
        # print("self.test_seen_label:", list(set(self.test_seen_label.numpy())))
        # print("self.seenclasses:", list(set(self.seenclasses.numpy())))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att


def default_flist_reader(opt, image_files, img_loc, image_labels, dataset):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if dataset == 'CUB':
            image_file = opt.image_root + image_file[0].split("MSc/")[1]
        elif dataset == 'AWA1':
            image_file = opt.image_root + image_file[0].split("databases/")[1]
        elif dataset == 'AWA2':
            image_file = opt.image_root + '/AwA2/JPEGImages/' + image_file[0].split("JPEGImages")[1]
        elif dataset == 'SUN':
            image_file = os.path.join(opt.image_root, image_file[0].split("data/")[1])
        else:
            exit(1)
        imlist.append((image_file, int(image_label)))
    return imlist


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, opt, data_inf=None, transform=None, target_transform=None, dataset=None,
                 flist_reader=default_flist_reader, loader=default_loader, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        if image_type == 'test_unseen_small_loc':
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'test_seen_loc':
            self.img_loc = data_inf.test_seen_loc
        elif image_type == 'trainval_loc':
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            self.img_loc = data_inf.train_loc
        else:
            try:
                sys.exit(0)
            except:
                print("choose the image_type in ImageFileList")

        if select_num != None:
            # select_num is the number of images that we want to use
            # shuffle the image loc and choose #select_num images
            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]

        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.dataset = dataset
        self.imlist = flist_reader(opt, self.image_files, self.img_loc, self.image_labels, self.dataset)

        self.imlist_backup = copy.deepcopy(self.imlist)
        self.allclasses_name = data_inf.allclasses_name
        self.attri_name = data_inf.attri_name

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.aug_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            if img.size == (224, 224):
                img = self.aug_transform(img)
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        num = len(self.imlist)
        return num


def get_loader(opt, data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(int(opt.size * 8. / 7.)),
        transforms.RandomCrop(opt.size),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomCrop(size=(64, 64)),
        # transforms.Scale((448, 448)),
        # transforms.RandomGrayscale(0.4),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        normalize,
        ])
    test_transform = transforms.Compose([
                                        transforms.Resize(opt.size),
                                        transforms.CenterCrop(opt.size),
                                        transforms.ToTensor(),
                                        normalize,])

    dataset_train = ImageFilelist(opt, data_inf=data, transform=train_transform,
                                  dataset=opt.dataset, image_type='trainval_loc')
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    dataset_test_unseen = ImageFilelist(opt, data_inf=data, transform=test_transform,
                                        dataset=opt.dataset, image_type='test_unseen_loc')
    testloader_unseen = torch.utils.data.DataLoader(dataset_test_unseen, batch_size=opt.batch_size, shuffle=False,
                                        num_workers=4, pin_memory=True)

    dataset_test_seen = ImageFilelist(opt, data_inf=data, transform=transforms.Compose([
                                                                    transforms.Resize(opt.size),
                                                                    transforms.CenterCrop(opt.size),
                                                                    transforms.ToTensor(),
                                                                    normalize, ]),
                                                          dataset=opt.dataset,
                                                          image_type='test_seen_loc')
    testloader_seen = torch.utils.data.DataLoader(dataset_test_seen, batch_size=opt.batch_size, shuffle=False,
                                        num_workers=4, pin_memory=True)

    return trainloader, testloader_unseen, testloader_seen
