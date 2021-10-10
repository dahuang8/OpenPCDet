import argparse
import glob
import os

import mayavi.mlab as mlab
import numpy as np

from visual_utils import visualize_utils as V

from file_utils.file_loaders import *

CFG = {'DATASET_PATH': '../datasets/kitti/'}


class Visualizer:
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path
        self._index = 0

    @staticmethod
    def get_all_indexes_from_path(dataset_path):
        indexes = list()
        for file in glob.glob(dataset_path + '/det_results/*.txt'):
            indexes.append(int(file.split('/')[-1].split('.')[0]))
        indexes.sort()
        print(indexes)
        return indexes

    def check_integrity(self, index):
        indexes = [index] if index >= 0 else self.get_all_indexes_from_path(
            self._dataset_path)

        for idx in indexes:
            # we gonna search training dataset
            # TODO(khuang): also search test dataset
            if idx >= 0:
                res_file_path = self._dataset_path + '/det_results/' + str(
                    idx) + '.txt'
                img_file_path = self._dataset_path + '/training/image_2/' + '{:06d}'.format(
                    idx) + '.png'
                gt_file_path = self._dataset_path + '/training/label_2/' + '{:06d}'.format(
                    idx) + '.txt'
                calib_file_path = self._dataset_path + '/training/calib/' + '{:06d}'.format(
                    idx) + '.txt'
                pc_file_path = self._dataset_path + '/training/velodyne/' + '{:06d}'.format(
                    idx) + '.bin'

                if not os.path.exists(res_file_path) or not os.path.exists(
                        img_file_path
                ) or not os.path.exists(gt_file_path) or not os.path.exists(
                        calib_file_path) or not os.path.exists(pc_file_path):
                    print('Integrity check fails for index: ' + str(idx))
                    return False
        return True

    def show_res(self, index):
        if self.check_integrity(index) == False:
            return
        print('Integrity test passed.')
        # display result for each index
        indexes = [index] if index >= 0 else self.get_all_indexes_from_path(
            self._dataset_path)
        for idx in indexes:
            res_file_path = self._dataset_path + '/det_results/' + str(
                idx) + '.txt'
            img_file_path = self._dataset_path + '/training/image_2/' + '{:06d}'.format(
                idx) + '.png'
            gt_file_path = self._dataset_path + '/training/label_2/' + '{:06d}'.format(
                idx) + '.txt'
            calib_file_path = self._dataset_path + '/training/calib/' + '{:06d}'.format(
                idx) + '.txt'
            gt_label = load_gt_file(gt_file_path)
            res_label = load_res_file(res_file_path)
            img = load_img(img_file_path)
            calib = load_calib_file(calib_file_path)
            pc = load_pc_file(pc_file_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument('--dataset_path',
                        type=str,
                        default=CFG['DATASET_PATH'],
                        help='Path to the dataset')
    parser.add_argument('--index',
                        type=int,
                        default=0,
                        help='Index of the image to be visualized')

    args = parser.parse_args()

    vis = Visualizer(args.dataset_path)
    vis.show_res(args.index)
