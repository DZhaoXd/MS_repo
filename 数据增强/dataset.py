import os
import numpy as np
import mindspore.dataset as ds
from src import load_UBC_for_train, load_UBC_for_test
import random

class TripletPhotoTour:
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """

    def __init__(self, data_root, dataset, sz_patch, num_pt_per_batch, nb_pat_per_pt, epoch_max, train=True, start_step = 0, sz_batch = 500):
        self.train = train
        self.patch = None
        self.pointID = None
        self.index = None
        self.start_step = start_step
        self.sz_batch = sz_batch
        if self.train:
            self.patch, self.pointID, self.index = load_UBC_for_train(data_root, dataset, sz_patch,
                               num_pt_per_batch, nb_pat_per_pt,
                               epoch_max)
        else:
            self.patch, self.pointID, self.index = load_UBC_for_test(data_root, dataset, sz_patch)
    def __getitem__(self, index):
        if self.train:
            index_batch = self.index[self.start_step // self.index.shape[1]][index]
            batch = self.patch[index_batch]
            self.start_step += 1
            return batch.transpose(1,0,2,3).astype(np.float32)
        else:
            nb_patch = self.pointID.size
            st = index * self.sz_batch
            en = np.min([(index + 1) * self.sz_batch, nb_patch])
            return self.patch[st:en].astype(np.float32)
    def __len__(self):
        if self.train:
            return self.index.shape[1]
        else:
            nb_patch = self.pointID.size
            return int(np.ceil(nb_patch / self.sz_batch))

class DataAugment:
    def __init__(self,
                 num_ID_per_batch):
        super(DataAugment, self).__init__()

        self.num_ID_per_batch = num_ID_per_batch

    def __call__(self, patch):
        for i in range(0, self.num_ID_per_batch):
            if random.random() > 0.5:
                nb_rot = np.random.randint(1, 4)
                patch[2 * i] = np.rot90(patch[2 * i], nb_rot)
                patch[2 * i + 1] = np.rot90(patch[2 * i + 1], nb_rot)

            if random.random() > 0.5:
                patch[2 * i] = np.flipud(patch[2 * i])
                patch[2 * i + 1] = np.flipud(patch[2 * i + 1])
        return patch[:, None, :, :]


def create_loaders(args=None):

    # For each train batch, we first sample #num_pt_per_batch 3D points, and then for each of the 3D point we sample #nb_pat_per_pt patches
    ubc_subset = ['yosemite', 'notredame', 'liberty']
    data_root = args.data_root
    train_set = args.train_set
    sz_patch = args.sz_patch
    epoch_max = args.epoch_max
    nb_pat_per_pt = args.nb_pat_per_pt
    num_pt_per_batch = args.num_pt_per_batch
    flag_dataAug = args.flag_dataAug
    test_set = []
    for val in ubc_subset:
        if val != train_set:
            test_set.append(val)
    #dataAugment
    train_triplet_aug_op = DataAugment(num_pt_per_batch)

    train_dataset = ds.GeneratorDataset(
        TripletPhotoTour(data_root, train_set,
                           sz_patch,
                           num_pt_per_batch, nb_pat_per_pt,
                           epoch_max),
        column_names=["patch"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.group_size)
    if flag_dataAug:
        train_dataset = train_dataset.map(
            operations=train_triplet_aug_op,
            input_columns=['patch'],
            num_parallel_workers=8)

    test_dataset = [ds.GeneratorDataset(
        TripletPhotoTour(data_root, val,
                         sz_patch,
                         num_pt_per_batch, nb_pat_per_pt,
                         epoch_max, train=False),
        column_names=["patch"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.group_size)
        for i, val in enumerate(test_set)]

    return train_dataset, test_dataset
