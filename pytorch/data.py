#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.data = self.data[np.random.choice(self.data.shape[0], min(self.data.shape[0],200), replace=False)]
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet10(Dataset):
    def __init__(self, partition='train', label_str_to_idx=None):
        self.vertices, self.faces, self.label, self.label_str_to_idx = self.load_OFF_data(partition, label_str_to_idx)
        self.label_idx_to_str = {v:k for k,v in self.label_str_to_idx.items()}
        # self.vertices = self.vertices[np.random.choice(len(self), min(len(self),200), replace=False)]
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.vertices[item]
        faces = self.faces[item]
        label = self.label[item]
        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        return pointcloud, faces, label

    def __len__(self):
        return len(self.vertices)

    def load_OFF_data(self, partition, label_str_to_idx=None):
        if label_str_to_idx is None:
            label_str_to_idx = {}
        next_idx = 0
        def read_off(fpath):
            with open(fpath, 'r') as file:
                if 'OFF' != file.readline().strip():
                    raise('Not a valid OFF header')

                n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
                verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
                faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
                return np.array(verts), np.array(faces)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        all_verts = []
        all_faces = []
        all_label = []
        for fpath in glob.glob(os.path.join(data_dir, 'ModelNet10', '*', partition, '*.off')):
            label_str = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            if label_str not in label_str_to_idx:
                label_str_to_idx[label_str] = next_idx
                next_idx += 1
            label = label_str_to_idx[label_str]

            verts, faces = read_off(fpath)
            all_verts.append(verts)
            all_faces.append(faces)
            all_label.append(label)

        return all_verts, all_faces, all_label, label_str_to_idx


if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for idx, (data, label) in enumerate(train):
    #     print(idx, data.shape, label)

    train = ModelNet10()
    test = ModelNet10('test', train.label_str_to_idx)
    for idx, (verts, faces, label) in enumerate(train):
        print(idx, verts.shape, verts[:3], faces.shape, faces[:3], label, train.label_idx_to_str[label])
