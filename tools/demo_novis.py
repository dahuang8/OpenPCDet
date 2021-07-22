import argparse
import glob
from pathlib import Path
import os
import math

# import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None,
                 ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(
            root_path /
            f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index],
                                 dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file',
                        type=str,
                        default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path',
                        type=str,
                        default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='specify the pretrained model')
    parser.add_argument(
        '--ext',
        type=str,
        default='.bin',
        help='specify the extension of your point cloud data file')
    parser.add_argument('--result_dir',
                        type=str,
                        default='/tmp/det_res',
                        help='specify the file to save the results')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def write_results(det, res_dir, class_names):
    print('{} results will be saved to {}'.format(len(det), res_dir))
    os.makedirs(res_dir, exist_ok=True)
    for i0, res in enumerate(det):
        file_path = os.path.join(res_dir, '{:06d}.txt'.format(i0))
        with open(file_path, 'w') as f:
            for i1 in range(len(res)):
                box = res[i1]['bbox3d']
                score = res[i1]['score'].item()
                cls = class_names[res[i1]['label'] - 1]
                # TODO(khuang): use camera calib
                # convert from normative to camera frame
                x_n = box[0].item()
                y_n = box[1].item()
                z_n = box[2].item()
                l_n = box[3].item()
                w_n = box[4].item()
                h_n = box[5].item()
                r_n = box[6].item()
                x_c = -y_n
                y_c = -z_n
                z_c = x_n
                r_c = (-r_n - math.pi/2 + math.pi) % (math.pi * 2) - math.pi
                alpha = np.arctan2(-x_c, z_c) # draw the frame, you'll understand this
                # write in kitti format
                f.write('{} 0.0 0 {:f} 0 0 0 0 {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}'.format(cls, alpha, h_n, w_n, l_n, x_c, y_c, z_c, r_c, score))
                f.write('\n')



def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG,
                               class_names=cfg.CLASS_NAMES,
                               training=False,
                               root_path=Path(args.data_path),
                               ext=args.ext,
                               logger=logger)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    results = list()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # note: these are in normative coordinates
            # it will be converted to camera frame in write_results
            ref_boxes = pred_dicts[0]['pred_boxes']
            ref_scores = pred_dicts[0]['pred_scores']
            ref_labels = pred_dicts[0]['pred_labels']

            # log this info as numpy and write to file at the end
            assert (len(ref_boxes) == len(ref_scores) == len(ref_labels))
            res_one_sample = list()
            for i in range(len(ref_boxes)):
                res_one_sample.append({
                    'bbox3d': ref_boxes[i],
                    'score': ref_scores[i],
                    'label': ref_labels[i]
                })

            results.append(res_one_sample)

            print('ref_boxes: {}\nref_scores: {}\nref_labels: {}'.format(
                ref_boxes, ref_scores, ref_labels))

            # V.visualize_prediction(pred_dicts[0], demo_dataset.class_names, demo_dataset.root_path)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)
    write_results(results, args.result_dir, demo_dataset.class_names)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
