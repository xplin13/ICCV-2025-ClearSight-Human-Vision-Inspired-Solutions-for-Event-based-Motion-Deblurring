import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
import glob
import cv2
import random


def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    """
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3]

    pols[pols == 0] = -1

    tis = ts.astype(np.int32)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def image_process(inp_img, inp_event, tar_img, ps, opt):
    """Process images for training with random crop and augmentation"""
    w, h = tar_img.shape[1], tar_img.shape[2]
    padw = ps - w if w < ps else 0
    padh = ps - h if h < ps else 0

    if padw != 0 or padh != 0:
        inp_img = TF.pad(torch.from_numpy(inp_img), (0, 0, padw, padh), padding_mode='reflect')
        tar_img = TF.pad(torch.from_numpy(tar_img), (0, 0, padw, padh), padding_mode='reflect')
        inp_event = TF.pad(torch.from_numpy(inp_event), (0, 0, padw, padh), padding_mode='reflect')
    else:
        inp_img = torch.from_numpy(inp_img)
        tar_img = torch.from_numpy(tar_img)
        inp_event = torch.from_numpy(inp_event)

    hh, ww = tar_img.shape[1], tar_img.shape[2]

    rr = random.randint(0, hh - ps)
    cc = random.randint(0, ww - ps)
    aug = random.randint(0, 8)

    # Crop patch
    inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
    tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
    inp_event = inp_event[:, rr:rr + ps, cc:cc + ps]

    # Data Augmentations
    if aug == 1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
        inp_event = inp_event.flip(1)
    elif aug == 2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
        inp_event = inp_event.flip(2)
    elif aug == 3:
        inp_img = torch.rot90(inp_img, dims=(1, 2))
        tar_img = torch.rot90(tar_img, dims=(1, 2))
        inp_event = torch.rot90(inp_event, dims=(1, 2))
    elif aug == 4:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        inp_event = torch.rot90(inp_event, dims=(1, 2), k=2)
    elif aug == 5:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        inp_event = torch.rot90(inp_event, dims=(1, 2), k=3)
    elif aug == 6:
        inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        inp_event = torch.rot90(inp_event.flip(1), dims=(1, 2))
    elif aug == 7:
        inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
        inp_event = torch.rot90(inp_event.flip(2), dims=(1, 2))

    return inp_img, inp_event, tar_img


class DataLoaderTrain_npz(Dataset):
    """Training data loader for npz format event data"""
    
    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain_npz, self).__init__()
        self.args = args
        self.blur_img_path = os.path.join(rgb_dir, 'blur')
        self.event_img_path = os.path.join(rgb_dir, 'event')
        self.sharp_img_path = os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))
        self.sequences_list = inp_files_dirs
        
        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        print(f'Total {len(self.sequences_list)} event sequences')

    def __len__(self):
        return len(self.sequences_list)

    def __getitem__(self, index):
        file_item = self.sequences_list[index]

        blur_image_name_list = sorted(glob.glob(os.path.join(self.blur_img_path, file_item, '*.png')))
        sharp_image_name_list = sorted(glob.glob(os.path.join(self.sharp_img_path, file_item, '*.png')))
        event_image_name_list = sorted(glob.glob(os.path.join(self.event_img_path, file_item, '*.npz')))
        
        blur_num_images = len(blur_image_name_list)

        start_index = random.randint(0, blur_num_images - 1)
        frame_index = start_index
        
        blur_img = cv2.imread(blur_image_name_list[frame_index])
        blur_img = np.float32(blur_img) / 255.0

        sharp_img = cv2.imread(sharp_image_name_list[frame_index])
        sharp_img = np.float32(sharp_img) / 255.0

        event = np.load(event_image_name_list[frame_index])
        if len(event['t']) == 0:
            event_div_tensor = np.zeros((self.args.num_bins, self.DVS_stream_height, self.DVS_stream_width))
        else:
            event_window = np.stack((event['t'], event['x'], event['y'], event['p']), axis=1)
            event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=self.args.num_bins,
                                             width=self.DVS_stream_width,
                                             height=self.DVS_stream_height)

        event_frame = np.float32(event_div_tensor)

        blur_img = blur_img.transpose([2, 0, 1])
        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img, event_frame, sharp_img = image_process(blur_img, event_frame, sharp_img,
                                                              self.args.patch_size, self.args)

        return blur_img, sharp_img, event_frame


class DataLoaderVal_npz(Dataset):
    """Validation data loader for npz format event data"""
    
    def __init__(self, rgb_dir, args):
        self.rgb_dir = rgb_dir
        blur_img_path = os.path.join(rgb_dir, 'blur')
        event_img_path = os.path.join(rgb_dir, 'event')
        sharp_img_path = os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(blur_img_path))

        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args = args

        self.seqs = inp_files_dirs
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            blur_img_lists = sorted(glob.glob(os.path.join(blur_img_path, self.seqs[i], '*.png')))
            event_lists = sorted(glob.glob(os.path.join(event_img_path, self.seqs[i], '*.npz')))
            gt_img_lists = sorted(glob.glob(os.path.join(sharp_img_path, self.seqs[i], '*.png')))
            seq_info['blur'] = blur_img_lists
            seq_info['event'] = event_lists
            seq_info['gt'] = gt_img_lists
            length_temp = len(blur_img_lists)
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)

    def __len__(self):
        return self.seqs_info['length']

    def __getitem__(self, idx):
        seq_idx, frame_idx = 0, 0

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length']
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length

        blur_img = cv2.imread(self.seqs_info[seq_idx]['blur'][frame_idx])
        blur_img = np.float32(blur_img) / 255.0
        blur_img = blur_img.transpose([2, 0, 1])

        event = np.load(self.seqs_info[seq_idx]['event'][frame_idx])
        num_bins = getattr(self.args, 'num_bins', 12) if hasattr(self, 'args') else 12
        if len(event['t']) == 0:
            event_div_tensor = np.zeros((num_bins, self.DVS_stream_height, self.DVS_stream_width))
        else:
            event_window = np.stack((event['t'], event['x'], event['y'], event['p']), axis=1)
            event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=num_bins,
                                             width=self.DVS_stream_width,
                                             height=self.DVS_stream_height)

        event_frame = np.float32(event_div_tensor)

        sharp_img = cv2.imread(self.seqs_info[seq_idx]['gt'][frame_idx])
        sharp_img = np.float32(sharp_img) / 255.0
        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)

        return blur_img, sharp_img, event_frame