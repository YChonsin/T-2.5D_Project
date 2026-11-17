import torch
import torch.utils.data as udata
from src.utils import *
import numpy as np

import h5py, glob
def save_h5(file_mode, append, output_fname, patch_size,
            compression, data, start_train, start_val, train_num, val_num):
    train_labels, train_masks, val_labels, val_masks = data
    with h5py.File(output_fname, file_mode) as h5_file:
        if append:
            # 先获取并检查全部数据集，旧文件缺少 train/mask 时直接报错。
            train_label = h5_file['train/label']
            train_mask = h5_file['train/mask']
            val_label = h5_file['val/label']
            val_mask = h5_file['val/mask']

            # 训练标签和掩码必须与追加位置、切片尺寸一致，并且允许扩容。
            for dataset in (train_label, train_mask):
                if dataset.shape != (start_train, *patch_size):
                    raise ValueError(f'{dataset.name}: existing shape does not match append position and patch_size')
                limit = dataset.maxshape[0]
                if limit is not None and start_train + train_num > limit:
                    raise ValueError(f'{dataset.name}: dataset cannot grow to the requested size')

            # 验证集执行相同检查；全部通过后才改变文件中的数据集尺寸。
            for dataset in (val_label, val_mask):
                if dataset.shape != (start_val, *patch_size):
                    raise ValueError(f'{dataset.name}: existing shape does not match append position and patch_size')
                limit = dataset.maxshape[0]
                if limit is not None and start_val + val_num > limit:
                    raise ValueError(f'{dataset.name}: dataset cannot grow to the requested size')

            train_label.resize(start_train + train_num, axis=0)
            train_mask.resize(start_train + train_num, axis=0)
            val_label.resize(start_val + val_num, axis=0)
            val_mask.resize(start_val + val_num, axis=0)
        else:
            # 标签和掩码形状相同，均为 (数量, 高, 宽)，首轴可以继续扩展。
            train_label = h5_file.create_dataset(
                'train/label', shape=(train_num, *patch_size), dtype=np.float32,
                maxshape=(None, *patch_size), compression=compression)
            train_mask = h5_file.create_dataset(
                'train/mask', shape=(train_num, *patch_size), dtype=np.float32,
                maxshape=(None, *patch_size), compression=compression)
            val_label = h5_file.create_dataset(
                'val/label', shape=(val_num, *patch_size), dtype=np.float32,
                maxshape=(None, *patch_size), compression=compression)
            val_mask = h5_file.create_dataset(
                'val/mask', shape=(val_num, *patch_size), dtype=np.float32,
                maxshape=(None, *patch_size), compression=compression)
        train_label[start_train:start_train + train_num] = np.asarray(train_labels, dtype=np.float32)
        train_mask[start_train:start_train + train_num] = np.asarray(train_masks, dtype=np.float32)

        val_label[start_val:start_val + val_num] = np.asarray(val_labels, dtype=np.float32)
        val_mask[start_val:start_val + val_num] = np.asarray(val_masks, dtype=np.float32)


def load_norm_pat(data_path: str, patch_size: tuple,
                  stride: tuple, key='', remove_zero: bool=True,
                  norm='none', from3d: bool=False):
    print(f"Input path: {data_path}")
    full_dataset = glob.glob(os.path.join(data_path, '*.mat'))
    patches = []
    for file_path in full_dataset:
        arr = load_data_auto(file_path, key=key)
        match norm:
            case 'absmax':
                arr = arr / (np.max(np.abs(arr)) or 1.)
            case 'log1p':
                arr = norm_by_log(arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
            case 'TPNN':
                arr, _, _ = norm_by_arcsinh_std(arr)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
            case 'none':
                pass

        if from3d:
            for i in range(arr.shape[2]):
                tmp_patches = Ar2Patch(arr[:, :, i], *patch_size, *stride, remove_zero)
                patches.extend(tmp_patches)
        else:
            tmp_patches = Ar2Patch(arr, *patch_size, *stride, remove_zero)
            patches.extend(tmp_patches)

    return patches


def load_norm_pat_3d(data_path: str, patch_size: tuple,
                  stride: tuple, key='', remove_zero: bool=True,
                  norm='none'):
    print(f"Input path: {data_path}")
    full_dataset = glob.glob(os.path.join(data_path, '*.mat'))
    patches = []
    for file_path in full_dataset:
        arr = load_data_auto(file_path, key=key)
        match norm:
            case 'absmax':
                arr = arr / (np.max(np.abs(arr)) or 1.)
            case 'log1p':
                arr = norm_by_log(arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
            case 'TPNN':
                arr, _, _ = norm_by_arcsinh_std(arr)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
            case 'none':
                pass

        tmp_patches = Ar2Patch_3d(arr, *patch_size, *stride, remove_zero, padmode='reflect')
        patches.extend(tmp_patches)

    return patches

def load_norm_pat_3d_train_val(data_path: str, patch_size: tuple,
                  stride: tuple, key='', remove_zero: bool=True,
                  norm='none'):
    print(f"Input path: {data_path}")
    full_files = glob.glob(os.path.join(data_path, '*.mat'))
    random.shuffle(full_files)
    val_num = len(full_files) // 5
    if val_num < 1:
        raise ValueError(f"Number of files are too small for generating validation patches.")

    train_files = full_files[:-val_num]
    val_files = full_files[-val_num:]

    train_patches = []
    for path in train_files:
        arr = load_data_auto(path, key=key)
        match norm:
            case 'absmax':
                arr = arr / (np.max(np.abs(arr)) or 1.)
            case 'log1p':
                arr = norm_by_log(arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
            case 'TPNN':
                arr, _, _ = norm_by_arcsinh_std(arr)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
            case 'none':
                pass

        tmp_patches = Ar2Patch_3d(arr, *patch_size, *stride, remove_zero, padmode='reflect')
        train_patches.extend(tmp_patches)

    val_patches = []
    for path in val_files:
        arr = load_data_auto(path, key=key)
        match norm:
            case 'absmax':
                arr = arr / (np.max(np.abs(arr)) or 1.)
            case 'log1p':
                arr = norm_by_log(arr)
            case 'log1pmax':
                arr, _ = norm_by_log_max(arr)
            case 'TPNN':
                arr, _, _ = norm_by_arcsinh_std(arr)
            case 'asinh':
                arr = norm_by_arcsinh(arr)
            case 'none':
                pass

        tmp_patches = Ar2Patch_3d(arr, *patch_size, *stride, remove_zero, padmode='reflect')
        val_patches.extend(tmp_patches)

    return train_patches, val_patches


class MyDataset(udata.Dataset):
    '''

    '''
    def __init__(self, dataset_fname='./data', istrain=True,
                 ):
        super().__init__()
        self.h5_file = h5py.File(dataset_fname, 'r')

        foldername = 'train' if istrain else 'val'
        self.labels = self.h5_file[f'{foldername}/label']
        self.masks = self.h5_file[f'{foldername}/mask']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        mask = self.masks[index]
        feature = label * mask
        # numpy -> torch
        feature = torch.from_numpy(np.array(feature)).float().unsqueeze(0)
        label = torch.from_numpy(np.array(label)).float().unsqueeze(0)
        # mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)

        # return feature, label, mask

        return feature, label
