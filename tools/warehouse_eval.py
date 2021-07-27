import argparse
import os
import sys
from shapely.geometry import shape, Polygon
import math
import numpy as np

# TODO(khuang): get camera pose from calib file
CAMERA_HEIGHT = 1.5  # in meter

CLASS_NAMES = ['forklift', 'pallet', 'agv']
CLASS_IOU_THRESHOLD = {'forklift': 0.7, 'pallet': 0.5, 'agv': 0.5}


def calculate_iou_2d(pred_box, gt_box):
    """
    This is a simple IOU evaluation based on the 2D bboxes projected to ground plane.
    """
    d_r = pred_box[-1]
    R_pred = np.array([[np.cos(d_r), -np.sin(d_r)], [np.sin(d_r),
                                                     np.cos(d_r)]])
    d_x = pred_box[0]
    d_y = pred_box[1]
    d_z = pred_box[2]
    d_h = pred_box[3]
    d_w = pred_box[4]
    d_l = pred_box[5]
    pb_corners = np.array([[-d_l / 2, -d_w / 2], [d_l / 2, -d_w / 2],
                           [d_l / 2, d_w / 2], [-d_l / 2, d_w / 2]])
    pb_corners = np.matmul(R_pred, pb_corners.T)
    pb_corners += np.array([d_x, d_y]).reshape(-1, 1)
    g_x = gt_box[0]
    g_y = gt_box[1]
    g_z = gt_box[2]
    g_h = gt_box[3]
    g_w = gt_box[4]
    g_l = gt_box[5]
    g_r = float(gt_box[6])
    R_det = np.array([[np.cos(g_r), -np.sin(g_r)], [np.sin(g_r), np.cos(g_r)]])
    gb_corners = np.array([[-g_l / 2, -g_w / 2], [g_l / 2, -g_w / 2],
                           [g_l / 2, g_w / 2], [-g_l / 2, g_w / 2]])
    gb_corners = np.matmul(R_det, gb_corners.T)
    gb_corners += np.array([g_x, g_y]).reshape(-1, 1)
    # intersection
    dt_poly = Polygon(pb_corners.T)
    gt_poly = Polygon(gb_corners.T)
    iou = dt_poly.intersection(gt_poly).area / gt_poly.area
    pb_center = np.mean(pb_corners, axis=1)
    gb_center = np.mean(gb_corners, axis=1)
    err_center = np.linalg.norm(pb_center - gb_center)
    err_rot = (d_r - g_r + math.pi/2) % math.pi - math.pi/2
    return (iou, pb_corners.T, gb_corners.T, err_center, err_rot)


class Evaluator(object):
    def __init__(self, dataset_dir, result_dir):
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir

    def evaluate(self):
        if not self._check_integrity():
            print('Cannot find enough data for evaluation.')
            return None
        all_tp = {c: 0 for c in CLASS_NAMES}
        all_fp = {c: 0 for c in CLASS_NAMES}
        all_fn = {c: 0 for c in CLASS_NAMES}
        tp_centers_error = {c: [] for c in CLASS_NAMES}
        tp_rot_error = {c: [] for c in CLASS_NAMES}
        for f in os.listdir(self.result_dir):
            if f.endswith('.txt'):
                res_id = int(f.split('.')[0])
                gt_res = self.load_gt_result('testing/label_2', res_id)
                # dt_res = self.load_det_result(res_id)
                dt_res = self.load_gt_result(self.result_dir, res_id)
                # tp, fn, fp
                tp = {c: 0 for c in CLASS_NAMES}
                fn = {c: 0 for c in CLASS_NAMES}
                fp = {c: 0 for c in CLASS_NAMES}
                for dt_item in dt_res:
                    found = False
                    for gt_item in gt_res:
                        if dt_item['label'] == gt_item['label']:
                            (iou, pb_corners, gb_corners, err_center,
                             err_rot) = calculate_iou_2d(
                                 dt_item['bbox3d'], gt_item['bbox3d'])
                            print('{} - {}: iou={}'.format(res_id, dt_item['label'], iou))
                            if iou >= CLASS_IOU_THRESHOLD[dt_item['label']]:
                                tp[dt_item['label']] += 1
                                tp_centers_error[dt_item['label']].append(err_center)
                                tp_rot_error[dt_item['label']].append(err_rot)
                                found = True
                                break
                    if not found:
                        fp[dt_item['label']] += 1
                        print(
                            '======== FP found for {} on label {} ========\nGT:\n{}\nDT:\n{}\n'
                            .format(res_id, dt_item['label'], pb_corners,
                                    gb_corners))
                for gt_item in gt_res:
                    found = False
                    for dt_item in dt_res:
                        if dt_item['label'] == gt_item['label']:
                            (iou, pb_corners, gb_corners, err_center,
                             err_rot) = calculate_iou_2d(
                                 dt_item['bbox3d'], gt_item['bbox3d'])
                            if iou >= CLASS_IOU_THRESHOLD[dt_item['label']]:
                                found = True
                                break
                    fn[gt_item['label']] += 1 if not found else 0

            for c in CLASS_NAMES:
                all_tp[c] += tp[c]
                all_fn[c] += fn[c]
                all_fp[c] += fp[c]

        # calculate precision, recall, f-measure
        precision = {c: 0 for c in CLASS_NAMES}
        recall = {c: 0 for c in CLASS_NAMES}
        fmeasure = {c: 0 for c in CLASS_NAMES}
        for c in CLASS_NAMES:
            if all_tp[c] == 0 and all_fp[c] == 0:
                precision[c] = 0
            else:
                precision[c] = all_tp[c] / (all_tp[c] + all_fp[c])
            if all_tp[c] == 0 and all_fn[c] == 0:
                recall[c] = 0
            else:
                recall[c] = all_tp[c] / (all_tp[c] + all_fn[c])

        return {
            'precision': precision,
            'recall': recall,
            'err_center': tp_centers_error,
            'err_rot': tp_rot_error
        }

    def _get_result_ids(self):
        result_ids = []
        for f in os.listdir(self.result_dir):
            if f.endswith('.txt'):
                result_ids.append(int(f.split('.')[0]))
        result_ids.sort()
        return result_ids

    def load_det_result(self, result_id):
        res = list()
        result_file = os.path.join(self.result_dir, str(result_id) + '.txt')
        with open(result_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            items = line.split(' ')
            assert len(items) == 9, 'Wrong format for result file.'
            res.append({
                'label': items[0],
                'score': float(items[1]),
                'bbox': [float(it) for it in items[2:9]]
            })
        return res

    def load_gt_result(self, label_folder, gt_id):
        res = list()
        gt_file = os.path.join(self.dataset_dir, label_folder,
                               '{:06d}'.format(gt_id) + '.txt')
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            assert len(items) == 16, 'Wrong format of gt label file.'
            h = float(items[8])
            w = float(items[9])
            l = float(items[10])
            x = float(items[13])
            y = -float(items[11])
            z = -float(items[
                12]) + h / 2 + CAMERA_HEIGHT  # Use camera pose from calib file
            rot_z = (-math.pi / 2 - float(items[14]) + math.pi) % (math.pi * 2) - math.pi
            res.append({
                'label': items[0],
                'truncation': float(items[1]),
                'occlusion': int(items[2]),
                'alpha': float(items[3]),
                'bbox2d': [float(item) for item in items[4:8]],
                'bbox3d': [x, y, z, h, w, l, rot_z]
            })
        return res

    def _check_integrity(self):
        gt_indexes = set()
        for f in os.listdir(os.path.join(self.dataset_dir, 'testing/label_2')):
            if f.endswith('.txt'):
                gt_indexes.add(int(f.split('.')[0]))
        for f in os.listdir(self.result_dir):
            if f.endswith('.txt'):
                res_id = int(f.split('.')[0])
                if not res_id in gt_indexes:
                    print('result_id {} is not found in gt'.format(res_id))
                    return False
        return True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir',
                        type=str,
                        help='folder of result files output by demo_novis.py')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='folder of the dataset corresponding to the test result')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    eval = Evaluator(args.dataset_dir, args.result_dir)
    eval_res = eval.evaluate()

    if eval_res is None:
        sys.exit(1)

    precision = eval_res['precision']
    recall = eval_res['recall']
    err_center = eval_res['err_center']
    err_rot = eval_res['err_rot']

    det_err_dir = os.path.join(args.result_dir, 'det_err')
    os.makedirs(det_err_dir, exist_ok=True)

    for c in CLASS_NAMES:
        err_center_file = os.path.join(det_err_dir, 'err_center_{}.csv'.format(c))
        err_rot_file = os.path.join(det_err_dir, 'err_rot_{}.csv'.format(c))
        np.savetxt(err_center_file, err_center[c])
        np.savetxt(err_rot_file, err_rot[c])

    for c in CLASS_NAMES:
        print('========== {} =========='.format(c))
        print('precision: {:f}'.format(precision[c]))
        print('recall: {:f}'.format(recall[c]))
