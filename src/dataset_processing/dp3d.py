# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as udata
from src.utils import *
import glob

import h5py
from src.dataset_processing.dcls import load_norm_pat_3d, save_h5

def prepare_data_seis_recon(data_path, output_fname,
                            patch_size: tuple,
                            stride: tuple,
                            misstype='random',
                            remove_zero=True,
                            start_train=0,
                            start_val=0,
                            key='',
                            train_num=10000,
                            val_num=1000,
                            norm='log1p',
                            compression=None,
                            **kwargs):

    patches = load_norm_pat_3d(data_path, patch_size, stride,
                               key, remove_zero, norm)
    total_num = train_num + val_num
    print(f'Total patches: {len(patches)}')
    # 随机选取total_num个patch
    if len(patches) <= total_num:
        raise ValueError("Number of patches not enough. Please increase the size of the entire data, or reduce "
                         "the required number of patches.")

    print(f"Randomly sampling {total_num} patches from {len(patches)} patches")
    patches = random.sample(patches, total_num)

    labels = []
    masks = []
    print("Generating complete-incomplete pairs")
    for p in patches:
        labels.append(p)
        _, mask = remove_traces_3d(p, 0.4, 0.6)
        masks.append(mask)

    indices = list(range(len(labels)))
    random.shuffle(indices)
    labels = [labels[i] for i in indices]
    masks = [masks[i] for i in indices]

    train_labels = labels[:train_num]
    train_masks = masks[:train_num]
    val_labels = labels[train_num:]
    val_masks = masks[train_num:]

    # 4. 准备四个 HDF5 数据集。start 为追加前的记录数，两者为 0 时覆盖新建。
    append = start_train != 0 or start_val != 0
    file_mode = 'w'
    if append:
        file_mode = 'r+'
    print(f"Saving data to h5 file {output_fname}")
    save_h5(file_mode, append, output_fname, patch_size, compression,
            (train_labels, train_masks, val_labels, val_masks),
            start_train, start_val, train_num, val_num)

    # 文件总数包括本次新增数量和追加前已有的记录。
    print(f'Total Training set: {start_train + train_num} pairs')
    print(f'Total Validation set: {start_val + val_num} pairs')
    print(f'Dataset saved to {output_fname}')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='recon_training')
    parser.add_argument("--tnum", type=int, default=6000,
                        help="The number of samples you want to generate for training.")
    parser.add_argument("--vnum", type=int, default=1500,
                        help="The number of samples you want to generate for validation.")
    parser.add_argument("--norm", type=str, default="absmax",
                        help="The normalization you want to apply to your data.  'absmax', 'TPNN', or 'none'")
    parser.add_argument("--df", type=str, default="kerry",
                        help="The dataset foldername.")
    parser.add_argument("--shape", type=int, nargs=3, default=0,
                        help="Source patch height and width.")
    parser.add_argument("--stride", type=int, nargs=3, default=0,
                        help="Source patch height and width.")
    opt = parser.parse_args()


    start_train = 0
    start_val = 0

    train_num = opt.tnum
    val_num = opt.vnum

    shape = tuple(opt.shape)
    stride = tuple(opt.stride)

    norm = opt.norm
    dname = opt.df

    shapename = f"{shape[0]}" if shape[0] == shape[1] == shape[2] else f"{shape[0]}{shape[1]}{shape[2]}"
    input_path = f'./data/{dname}/train/'
    n_patches = round(train_num / 1000., 1) if train_num % 1000 != 0 \
        else round(train_num / 1000)
    output_fname = f"./dataset/{dname}_3D_{shapename}_{str(n_patches)}k.h5"


    os.makedirs("./dataset/", exist_ok=True)

    prepare_data_seis_recon(input_path, output_fname, shape, stride,
                                    remove_zero=False, misstype='random',
                                    key='', train_num=train_num, val_num=val_num,
                                    start_train=start_train, start_val=start_val,
                                    norm=norm, compression=None )

    from src.dataset_processing.dcls import MyDataset
    ds = MyDataset(dataset_fname=output_fname, istrain=True)
    dl = udata.DataLoader(ds, batch_size=1, shuffle=True)

    feature, label = next(iter(dl))
    # print(feature)
    # print(label)
    print(feature.shape)
    print(label.shape)
