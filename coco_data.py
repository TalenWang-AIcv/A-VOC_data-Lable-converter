# use coding;utf-8 import
import os
import sys
import shutil
import random
import pickle
import glob

import torch
from pycocotools.coco import COCO
import numpy as np
import PIL.Image
import torch.utils.data
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm

original_imgs_path = '../../../Datasets/COCO/'


class COCO_data(torch.utils.data.Dataset):
    def __init__(self, file_path, train=True, transform=None, target_transform=None):
        self.file_path = file_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not (os.path.isfile(os.path.join(self.file_path, 'train.pkl'))
                and os.path.isfile(os.path.join(self.file_path, 'test.pkl'))):
            self.process()

        if self.train:
            print('读取训练集数据...')
            self.train_data, self.train_labels = pickle.load(open(os.path.join(self.file_path, 'train.pkl'), 'rb'))
            print('读取成功！')
        else:
            print('读取测试集数据...')
            self.test_data, self.test_labels = pickle.load(
                open(os.path.join(self.file_path, 'test.pkl'), 'rb'))
            print('读取成功！')

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_data[index], self.train_labels[index]
        else:
            image, label = self.test_data[index], self.test_labels[index]

        image = PIL.Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def process(self):
        train_image_path = glob.glob(os.path.join(self.file_path, original_imgs_path + 'train2014_select20000/*.jpg'))
        test_image_path = glob.glob(os.path.join(self.file_path, original_imgs_path + 'val2017/*.jpg'))

        # 获取训练集样本的出席对象向量
        train_id_and_isAbsence_path = glob.glob(os.path.join(self.file_path, original_imgs_path + 'txt_train2014_select20000/*.txt'))
        np_train_labels = []
        for i, item in enumerate(train_id_and_isAbsence_path):
            train_id_and_isAbsence = list((np.genfromtxt(item))[:, 1])

            for i in range(len(train_id_and_isAbsence)):
                if train_id_and_isAbsence[i] == -1:
                    train_id_and_isAbsence[i] = 0
            np_train_labels.append(train_id_and_isAbsence)
        np_train_labels = np.transpose(np.array(np_train_labels))

        # 获取测试集样本的出席对象向量
        test_id_and_isAbsence_path = glob.glob(os.path.join(self.file_path, original_imgs_path + 'txt_val2017/*.txt'))
        np_test_labels = []
        for i, item in enumerate(test_id_and_isAbsence_path):
            test_id_and_isAbsence = list((np.genfromtxt(item))[:, 1])

            for i in range(len(test_id_and_isAbsence)):
                if test_id_and_isAbsence[i] == -1:
                    test_id_and_isAbsence[i] = 0
            np_test_labels.append(test_id_and_isAbsence)
        np_test_labels = np.transpose(np.array(np_test_labels))

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        print('数据预处理...')
        pbar = tqdm(total=(len(train_image_path) + len(test_image_path)))
        for i in range(len(train_image_path)):
            image = PIL.Image.open(train_image_path[i])
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            np_image = np.array(image)
            image.close()
            train_data.append(np_image)
            train_labels.append(np_train_labels[i, :])
            pbar.update(1)

        for i in range(len(test_image_path)):
            image = PIL.Image.open(test_image_path[i])
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            np_image = np.array(image)
            image.close()
            test_data.append(np_image)
            test_labels.append(np_test_labels[i, :])
            pbar.update(1)
        pbar.close()

        print('处理完成，存储文件...')
        pickle.dump((train_data, train_labels), open(os.path.join(self.file_path, 'train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels), open(os.path.join(self.file_path, 'test.pkl'), 'wb'))
        print('文件存储完成，正在进行下一步，请耐心等待...')


def demo():
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    dataDir = '../../Datasets/COCO/train2014'
    annoFile = '../../Datasets/COCO/annotations2014/instances_train2014.json'
    # Initial annotations data COCO API
    coco = COCO(annoFile)

    # Display COCO categories and super categories
    categories = coco.loadCats(coco.getCatIds())

    # get all images containing given category
    category_Ids = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=category_Ids)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    I = io.imread('%s/%s' % (dataDir, img['file_name']))
    plt.axis('off')
    plt.imshow(I)
    annoIds = coco.getAnnIds(imgIds=img['id'], catIds=category_Ids, iscrowd=None)
    annos = coco.loadAnns(annoIds)
    coco.showAnns(annos)
    plt.show()


def coco_xml2txt(xml_path, img_path, txt_path):
    # xml_path = '../../Datasets/COCO/instance_train_annotation_2014'
    # txt_path = '../../Datasets/COCO/txt_train2014'
    # img_path = '../../Datasets/COCO/train2014'
    xmls = []

    for category in tqdm((os.listdir(xml_path))):
        child_path = os.path.join(xml_path, category)
        for xml in (os.listdir(child_path)):
            xml = str(xml.split('.')[0])        # A category contains this img name
            xmls.append(xml)
        for img in (os.listdir(img_path)):
            img = str(img.split('.')[0])  # all img names
            file = open('%s/%s.txt' % (txt_path, category), 'a')
            if img in xmls:
                file.write('%s 1 \n' % img)
            else:
                file.write('%s -1 \n' % img)
        xmls = []


def random_select_half(img_path, select_img_path):
    imgs = []

    for img in (os.listdir(img_path)):
        imgs.append(img)

    for i in tqdm(range(20000)):
        select_img = random.randint(0, (len(imgs)-1))
        shutil.copyfile(img_path + imgs[select_img], select_img_path + imgs[select_img])


if __name__ == '__main__':
    img_path = '../../Datasets/COCO/train2014/'
    xml_path = '../../Datasets/COCO/instance_train_annotation_2014'

    select_img_path = '../../Datasets/COCO/train2014_select20000/'
    select_img_txt_path = '../../Datasets/COCO/txt_train2014_select20000'
    pkl_save_path = '../Data/COCO/'
    # random_select_half(img_path, select_img_path)
    # coco_xml2txt(xml_path, select_img_path, select_img_txt_path)
    coco = COCO_data(pkl_save_path)
