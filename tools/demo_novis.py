import argparse
import glob
from pathlib import Path
import os

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

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def write_det_res_kitti(det_res, file_path):
    pass

def main():
    OUTPUT_FOLDER = './det_results'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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
    test_results = list()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            ref_boxes = pred_dicts[0]['pred_boxes']
            ref_scores = pred_dicts[0]['pred_scores']
            ref_labels = pred_dicts[0]['pred_labels']

            with open(os.path.join(OUTPUT_FOLDER, '{}.txt'.format(idx)),
                      'w') as f:
                for i0, box in enumerate(ref_boxes):
                    score = ref_scores[i0].item()
                    label = ref_labels[i0].item()
                    corners = [str(x.item()) for x in box]
                    f.write('{} {} {}\n'.format(label, score,
                                                ' '.join(corners)))

            print('ref_boxes: {}\nref_scores: {}\nref_labels: {}'.format(
                ref_boxes, ref_scores, ref_labels))

            # V.visualize_prediction(pred_dicts[0], demo_dataset.class_names, demo_dataset.root_path)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
