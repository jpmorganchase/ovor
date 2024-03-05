# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import errno
import hashlib
import numpy as np
from os import makedirs
from os.path import exists, expanduser, isfile, join
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import yaml

class DatasetLoader:
    def __init__(self, root):
        self.root = expanduser(root)
        self.data = dict()
        self.load()

    def load(self):
        pass

class CIFAR100Loader(DatasetLoader):
    num_cls = 100
    input_size = [32, 32, 3]
    base_folder = 'cifar-100-python'
    
    def load(self):
        train_path = join(self.root, self.base_folder, 'train')
        test_path = join(self.root, self.base_folder, 'test')
        if not exists(train_path) or not exists(test_path):
            self._download()
        self.data['train'] = self._load_set(train_path)
        self.data['test'] = self._load_set(test_path)

    def _load_set(self, path):
        with open(path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        data = entry['data']
        targets = entry['labels' if 'labels' in entry else 'fine_labels']
        data = np.vstack(data).reshape(-1, self.input_size[2], self.input_size[0], self.input_size[1])
        data = data.transpose((0, 2, 3, 1))
        return np.asarray(data), np.asarray(targets)

    def _download(self):
        import tarfile
        filename = 'cifar-100-python.tar.gz'
        download_url(
            'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            self.root, filename,
            'eb9058c3a382ffc7106e4002c42a8d85'
        )
        with tarfile.open(join(self.root, filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

    def get_transform(self, mode='train'):
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        if mode == 'train':
            transform_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transform_list = [transforms.Resize(224)]
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)

class ImagenetRLoader(DatasetLoader):
    num_cls = 200
    input_size = [224, 224, 3]
    base_folder = 'imagenet-r'
    
    def load(self):
        cfg = yaml.load(open('dataloaders/splits/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        self.data['train'] = (np.asarray(cfg['data']), np.asarray(cfg['targets']))
        cfg = yaml.load(open('dataloaders/splits/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        self.data['test'] = (np.asarray(cfg['data']), np.asarray(cfg['targets']))

    def get_transform(self, mode='train'):
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        if mode == 'train':
            transform_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transform_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)

class CUBLoader(DatasetLoader):
    num_cls = 200
    base_folder = 'CUB_200_2011'

    def load(self):
        paths = self._load_file(join(self.root, self.base_folder, 'images.txt'))
        split = self._load_file(join(self.root, self.base_folder, 'train_test_split.txt'))
        targets = self._load_file(join(self.root, self.base_folder, 'image_class_labels.txt'))
        path_train, path_test = [], []
        targets_train, targets_test = [], []
        for p, s, t in zip(paths, split, targets):
            path = join(self.root, self.base_folder, 'images', p)
            if int(s) == 1:
                path_train.append(path)
                targets_train.append(int(t) - 1)
            else:
                path_test.append(path)
                targets_test.append(int(t) - 1)
        self.data['train'] = (np.array(path_train), np.array(targets_train))
        self.data['test'] = (np.array(path_test), np.array(targets_test))

    def _load_file(self, path):
        d = []
        with open(path) as f:
            for line in f:
                _, item = line.strip().split(' ')
                d.append(item)
        return d

    def get_transform(self, mode='train'):
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        if mode == 'train':
            transform_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transform_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)

class ImagenetALoader(DatasetLoader):
    num_cls = 200
    base_folder = 'imagenet-a'

    def load(self):
        dst_train = datasets.ImageFolder(join(self.root, self.base_folder, 'train'))
        self.data['train'] = (np.array([i[0] for i in dst_train.imgs]), np.array(dst_train.targets))
        dst_test = datasets.ImageFolder(join(self.root, self.base_folder, 'test'))
        self.data['test'] = (np.array([i[0] for i in dst_test.imgs]), np.array(dst_test.targets))

    def get_transform(self, mode='train'):
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        if mode == 'train':
            transform_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transform_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)

class ClassShuffledDataset(Dataset):
    def __init__(self, data, targets, cls_mapping, transform):
        self.data = data
        self.targets = targets
        self.cls_mapping = cls_mapping
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self._read_img(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_mapping[target], index

    def __len__(self):
        return len(self.data)

    def _read_img(self, img):
        pass

class CIFAR100(ClassShuffledDataset):
    def _read_img(self, img):
        return Image.fromarray(img)

class ImagenetR(ClassShuffledDataset):
    def _read_img(self, img):
        with Image.open(img) as i:
            i = i.convert('RGB')
            i_arr = np.fromstring(i.tobytes(), dtype=np.uint8)
            i_arr = i_arr.reshape((i.size[1], i.size[0], 3))
        return Image.fromarray(i_arr)

class CUB200(ClassShuffledDataset):
    def _read_img(self, img):
        with Image.open(img) as i:
            i = i.convert('RGB')
            i_arr = np.frombuffer(i.tobytes(), dtype=np.uint8)
            i_arr = i_arr.reshape((i.size[1], i.size[0], 3))
        return Image.fromarray(i_arr)

class ImagenetA(ClassShuffledDataset):
    def _read_img(self, img):
        with Image.open(img) as i:
            i = i.convert('RGB')
            i_arr = np.frombuffer(i.tobytes(), dtype=np.uint8)
            i_arr = i_arr.reshape((i.size[1], i.size[0], 3))
        return Image.fromarray(i_arr)

def check_integrity(fpath, md5):
    if not isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    fpath = join(root, filename)
    try:
        makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if not isfile(fpath) or not check_integrity(fpath, md5):
        try:
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                urllib.request.urlretrieve(url, fpath)