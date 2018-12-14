# -*- use coding:utf-8  -*-
import os
from shutil import copyfile
import numpy as np
from tqdm import tqdm

Txt_path = '..\VOC07train\ImageSets'
Img_path = '..\VOC07train\JPEGImages'
Out_file_path = '..\\negative'
no_keyworld_list = []
flag = 0


class SearchFile(object):
    def __init__(self, org_txt_path='..', img_path='..', out_path='..', keyworlds='person', lable='-1'):
        self.keyworlds = keyworlds
        self.lable = lable
        self.txt_path = org_txt_path
        self.txt_abspath = os.path.abspath(self.txt_path)  # 默认当前目录
        self.img_path = img_path
        self.img_abspath = os.path.abspath(self.img_path)  # 默认当前目录
        self.txt_out_path = out_path
        self.txt_out_abspath = os.path.abspath(self.txt_out_path)  # 默认当前目录
        self.img_out_path = out_path + '\\no_%s_imgs\\' % self.keyworlds
        self.img_out_abspath = os.path.abspath(self.img_out_path)  # 默认当前目录

    def mkDir(self):
        # 去除首位空格
        path = self.img_out_path.strip()
        # 去除尾部 / 符号
        path = path.rstrip("/")
        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            print(path + ' 创建成功')
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(path + ' 目录已存在')
            return False

    # 读取原始Txt文件下的数据，取出不包含关键字的文件名，并转换成训练所需格式，保存为.txt文件
    def Translate_txt_File(self):
        for child_dir in (os.listdir(self.txt_path)):
            child_path = os.path.join(self.txt_path, child_dir)
            for textfiles in os.listdir(child_path):
                if self.keyworlds in textfiles:
                    print(textfiles)
                    keyworld_txt_file = np.loadtxt(os.path.join(self.txt_path, child_dir) + '\\' + textfiles, dtype=np.str_)
                    for keyworld in keyworld_txt_file:
                        if self.lable in keyworld:
                                no_person_imgs = keyworld[0]+'.jpg'
                                if self.lable == '-1':
                                    no_keyworld_list.append([no_person_imgs, 0])
                                if self.lable == '1':
                                    no_keyworld_list.append([no_person_imgs, 1])
        print('%s targets have been find' % self.keyworlds)
        print('-----------Rewriting Target .txt file------------\n')
        no_keyworld_file = os.path.join(self.txt_out_path) + '\\no_%s.txt' % self.keyworlds
        output = open(no_keyworld_file, 'w', encoding='gbk')
        for i in tqdm(range(len(no_keyworld_list))):
            for j in range(len(no_keyworld_list[i])):
                output.write(str(no_keyworld_list[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
                output.write(' ')  # 相当于Tab一下，换一个单元格
            output.write('\n')  # 写完一行立马换行
        output.close()

        print(' File progress finish!')

    def Translate_Imgs(self, target_img_txt='..'):
        self.mkDir()
        Target_img = np.loadtxt(target_img_txt, dtype=np.str_)[:, 0]
        for i in tqdm(range(len(Target_img))):
            if Target_img[i] in os.listdir(self.img_path):
                imgs_path = os.path.join(self.img_path, Target_img[i])
                copyfile(imgs_path, self.img_out_path + Target_img[i])
        print('\n Target imgs have been copied to %s' % self.img_out_path)


if __name__ == '__main__':
    search = SearchFile(Txt_path, Img_path, Out_file_path, keyworlds='person_trainval', lable='-1')
    search.Translate_txt_File()
    # search.Translate_Imgs(target_img_txt='..\\negative\\no_person_trainval.txt')



