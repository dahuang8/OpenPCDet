from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos

if __name__ == '__main__':
  DATASET_CFG = 'tools/cfgs/dataset_configs/warehouse_kitti_dataset.yaml'
  CLASS_NAME = ['forklift', 'agv', 'pallet']
  DATA_PATH = './data/kitti'
  SAVE_PATH = '/root/kitti'
  TRAINING = True
  create_kitti_infos(dataset_cfg=DATASET_CFG, class_names=CLASS_NAME, data_path=DATA_PATH, save_path=SAVE_PATH)
