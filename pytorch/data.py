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
import pickle
import tqdm
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


def read_OFF(fpath):
    with open(fpath, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')

        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return np.array(verts, dtype='float32'), np.array(faces, dtype='int64')


def modelnet10_OFF_preproc(partition, out_pkl_path=None, mn10_dir=None):
    if mn10_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mn10_dir = os.path.join(base_dir, 'data', 'ModelNet10')

    if out_pkl_path is None:
        out_pkl_path = os.path.join(mn10_dir, f'preprocessed_data__{partition}.pkl')
    if os.path.isfile(out_pkl_path):
        print(f"Reusing preprocessed file: {out_pkl_path}")
        return out_pkl_path  # don't change existing preprocessed data


    print(f"Creating data file: {out_pkl_path}")
    data = {key: [] for key in ["vertices", "faces", "label"]}
    label_str_to_idx = {}
    next_idx = 0
    for basename in sorted(os.listdir(mn10_dir)):   # sort to have fixed order of labels
        dirpath = os.path.join(mn10_dir, basename)
        if os.path.isdir(dirpath):
            label_str_to_idx[basename] = np.array(next_idx, dtype='int64')
            next_idx += 1

            for fpath in tqdm.tqdm(
                    glob.glob(os.path.join(dirpath, partition, '*.off')),
                    desc=f"Label: {basename}"
                ):
                vertices, faces = read_OFF(fpath)
                data["vertices"].append(vertices)
                data["faces"].append(faces)
                data["label"].append(label_str_to_idx[basename])

    data['label_str_to_idx'] = label_str_to_idx
    data['label_idx_to_str'] = {v:k for k,v in label_str_to_idx.items()}

    with open(out_pkl_path, 'wb') as pf:
        pickle.dump(data, pf)

    return out_pkl_path


class ModelNet10(Dataset):
    def __init__(self, num_points, partition='train', mn10_dir=None, pkl_path=None):
        pkl_path = modelnet10_OFF_preproc(partition, out_pkl_path=pkl_path, mn10_dir=mn10_dir)

        with open(pkl_path, 'rb') as pf:
            data = pickle.load(pf)
            self.vertices, self.faces, self.label = data["vertices"], data["faces"], data["label"]
            self.label_str_to_idx, self.label_idx_to_str = data['label_str_to_idx'], data['label_idx_to_str']

        # self.vertices = self.vertices[np.random.choice(len(self), min(len(self),200), replace=False)]
        self.partition = partition
        self.num_points = num_points        

    def __getitem__(self, item):
        pointcloud = self.vertices[item][:self.num_points].astype('float32')
        faces = self.faces[item]
        label = self.label[item]
        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        return pointcloud, faces, label

    def __len__(self):
        return len(self.vertices)


def read_OBJ(fpath):
    with open(fpath, 'r') as file:
        verts = []
        faces = []
        for line in file:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue

            lsp = line.split()
            if lsp[0] == 'v':
                verts.append([float(w) for w in lsp[1:]])
            elif lsp[0] == 'f':
                faces.append([(int(w)-1) for w in lsp[1:]])
        return np.array(verts, dtype='float32'), np.array(faces, dtype='int64')


def shrec16_OBJ_preproc(partition, out_pkl_path=None, data_root_dir=None):
    if data_root_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_root_dir = os.path.join(base_dir, 'data', 'shrec_16')

    if out_pkl_path is None:
        out_pkl_path = os.path.join(data_root_dir, f'preprocessed_data__{partition}.pkl')
    if os.path.isfile(out_pkl_path):
        print(f"Reusing preprocessed file: {out_pkl_path}")
        return out_pkl_path  # don't change existing preprocessed data


    print(f"Creating data file: {out_pkl_path}")
    data = {key: [] for key in ["vertices", "faces", "label"]}
    label_str_to_idx = {}
    next_idx = 0
    for basename in sorted(os.listdir(data_root_dir)):   # sort to have fixed order of labels
        dirpath = os.path.join(data_root_dir, basename)
        if os.path.isdir(dirpath):
            label_str_to_idx[basename] = np.array(next_idx, dtype='int64')
            next_idx += 1

            for fpath in tqdm.tqdm(
                    glob.glob(os.path.join(dirpath, partition, '*.obj')),
                    desc=f"Label: {basename}"
                ):
                vertices, faces = read_OBJ(fpath)
                data["vertices"].append(vertices)
                data["faces"].append(faces)
                data["label"].append(label_str_to_idx[basename])

    data['label_str_to_idx'] = label_str_to_idx
    data['label_idx_to_str'] = {v:k for k,v in label_str_to_idx.items()}

    with open(out_pkl_path, 'wb') as pf:
        pickle.dump(data, pf)

    return out_pkl_path


class SHREC16(Dataset):
    def __init__(self, num_points, partition='train', data_root_dir=None, pkl_path=None):
        pkl_path = shrec16_OBJ_preproc(partition, out_pkl_path=pkl_path, data_root_dir=data_root_dir)

        with open(pkl_path, 'rb') as pf:
            data = pickle.load(pf)
            self.vertices, self.faces, self.label = data["vertices"], data["faces"], data["label"]
            self.label_str_to_idx, self.label_idx_to_str = data['label_str_to_idx'], data['label_idx_to_str']

        # self.vertices = self.vertices[np.random.choice(len(self), min(len(self),200), replace=False)]
        self.partition = partition
        self.num_points = num_points        

    def __getitem__(self, item):
        pointcloud = self.vertices[item][:self.num_points].astype('float32')
        # faces = self.faces[item]
        label = self.label[item]
        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        # return pointcloud, faces, label
        return pointcloud, label

    def __len__(self):
        return len(self.vertices)


if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for idx, (data, label) in enumerate(train):
    #     print(idx, data.shape, label)

    # print("=== TRAIN: ModelNet10 ===")
    # train = ModelNet10(partition='train', num_points=1024)
    # idxs = np.random.choice(len(train), 10)
    # for idx in idxs:
    #     vertices, faces, label = train[idx]
    #     print(idx, label, train.label_idx_to_str[label], '\n',
    #     vertices.shape, vertices[:3], '\n',
    #     faces.shape, faces[:3])

    # print("=== TEST: ModelNet10 ===")
    # test = ModelNet10(partition='test', num_points=1024)
    # idxs = np.random.choice(len(test), 10)
    # for idx in idxs:
    #     vertices, faces, label = test[idx]
    #     print(idx, label, test.label_idx_to_str[label], '\n',
    #     vertices.shape, vertices[:3], '\n',
    #     faces.shape, faces[:3])

    print("=== TRAIN: SHREC16 ===")
    train = SHREC16(partition='train', num_points=1024)
    idxs = np.random.choice(len(train), 10)
    for idx in idxs:
        vertices, label = train[idx]
        print(idx, label, train.label_idx_to_str[label], '\n',
        vertices.shape, vertices[:3])

    print("=== TEST: SHREC16 ===")
    test = SHREC16(partition='test', num_points=1024)
    idxs = np.random.choice(len(test), 10)
    for idx in idxs:
        vertices, label = test[idx]
        print(idx, label, test.label_idx_to_str[label], '\n',
        vertices.shape, vertices[:3])

    # vertices, faces, label = test[0]
    # with open('test.off', 'w') as file:
    #     file.write('OFF\n')
    #     file.write(f'{len(vertices)} {len(faces)} 0\n')
    #     file.writelines([
    #         ( ' '.join([str(coord) for coord in v]) + '\n' )
    #         for v in vertices
    #     ])
    #     file.writelines([
    #         ( str(len(f)) + ' ' + ' '.join([str(idx) for idx in f]) + '\n' )
    #         for f in faces
    #     ])
