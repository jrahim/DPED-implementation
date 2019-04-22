from glob import glob
from multiprocessing import Pool
from math import ceil
from utils import *
import time
import os
import scipy


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.mean = None
        self.std = None
        self.phone = config.phone_model
        if config.test_mode:
            if config.test_patches:
                self.mode = "test_data/patches"
            else:
                self.mode = "test_data/full_size_test_images"
        else:
            self.mode = "training_data"
        self.phone_data, self.dslr_data, self.width, self.height = self.load_data()

    def load_data(self):
        if self.config.run_img:
            phone_files = glob(os.path.join(self.config.run_img, "*"))
        elif (not self.config.num_files_to_load) and self.mode == "training_data":
            phone_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*")))
            dslr_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "canon/*")))
        elif (not self.config.num_files_to_load) and self.mode == "test_data/patches":
            print("test files loading: ", os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*"))
            phone_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*")))
            dslr_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "canon", "*")))
        elif self.config.num_files_to_load and self.mode == "test_data/patches":
            print("test files loading: ",
                  os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*"))
            phone_files = sorted(
                glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*")))[
                          :self.config.num_files_to_load]
            dslr_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "canon", "*")))[
                         :self.config.num_files_to_load]
        elif (not self.config.num_files_to_load) and self.mode == "test_data/full_size_test_images":
            print("test files loading: ", os.path.join(self.config.dataset_dir, self.phone, self.mode, "*"))
            phone_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "*")))
        elif self.config.num_files_to_load and self.mode == "test_data/full_size_test_images":
            print("test files loading: ",
                  os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*"))
            phone_files = sorted(
                glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "*")))[
                          :self.config.num_files_to_load]
        else:
            phone_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, self.phone, "*")))[
                          :self.config.num_files_to_load]
            dslr_files = sorted(glob(os.path.join(self.config.dataset_dir, self.phone, self.mode, "canon/*")))[
                         :self.config.num_files_to_load]
        print("number of total files to be loaded: ", len(phone_files))

        start_time = time.time()
        pool = Pool(processes=8)
        train_num = int(ceil(len(phone_files) / 8))

        # Load data
        phone_loaders = [
            pool.apply_async(load_files, (
                phone_files[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
            for i in range(8)]
        phone_data = []
        for res in phone_loaders:
            phone_data.extend(res.get())

        if (self.mode == "training_data" or self.mode == "test_data/patches") and not self.config.run_img:
            dslr_loaders = [
                pool.apply_async(load_files, (
                    dslr_files[i * train_num:i * train_num + train_num], self.config.res, self.config.test_mode))
                for i in range(8)]
            dslr_data = []
            for res in dslr_loaders:
                dslr_data.extend(res.get())
        else:
            dslr_data = []

        time2 = time.time() - start_time
        print("%d image pairs loaded for training set! setting took: %4.4fs" % (len(phone_data), time2))

        width = len(phone_data[0])
        height = len(phone_data[0][0])

        # standardize input images
        # self.mean = np.mean(noisy_train, axis=(1, 2), keepdims=True)
        # self.std = np.std(noisy_train, axis=(1, 2), keepdims=True)
        # noisy_train = (noisy_train - self.mean) / self.std
        # noisy_test = (noisy_test - self.mean) / self.std

        return phone_data, dslr_data, width, height

    def get_batch(self):
        phone_batch = np.zeros(
            [self.config.batch_size, self.width, self.height, 3],
            dtype='float32')
        dslr_batch = np.zeros(
            [self.config.batch_size, self.width, self.height, 3],
            dtype='float32')

        for i in range(self.config.batch_size):
            index = np.random.randint(len(self.phone_data))
            phone_patch = self.phone_data[index]
            dslr_patch = self.dslr_data[index]

            # randomly flip, rotate patch (assuming that the patch shape is square)
            if self.config.augment:
                prob = np.random.rand()
                if prob > 0.5:
                    phone_patch = np.flip(phone_patch, axis=0)
                    dslr_patch = np.flip(dslr_patch, axis=0)
                prob = np.random.rand()
                if prob > 0.5:
                    phone_patch = np.flip(phone_patch, axis=1)
                    dslr_patch = np.flip(dslr_patch, axis=1)
                prob = np.random.rand()
                if prob > 0.5:
                    phone_patch = np.rot90(phone_patch)
                    dslr_patch = np.rot90(dslr_patch)
            phone_batch[i, :, :, :] = preprocess(phone_patch)  # pre/post processing function is defined in utils.py
            dslr_batch[i, :, :, :] = preprocess(dslr_patch)
        return phone_batch, dslr_batch
