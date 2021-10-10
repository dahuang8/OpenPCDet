import struct
import numpy as np

def load_pc_file(file_path):
    list_pcd = list()
    SIZE_FLOAT = 4
    with open(file_path, "rb") as f:
        byte = f.read(SIZE_FLOAT * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(SIZE_FLOAT * 4)
    np_pcd = np.asarray(list_pcd)
    return np_pcd


def load_gt_file(gt_file_path):
    # output will be label, bbox2d and bbox3d
    with open(gt_file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    labels = list()
    for l in lines:
        entry = dict()
        items = l.split(' ')
        entry['label'] = int(items[0])
        bbox2d = {'x_min': float(items[3]), 'y_min': float(items[4]), 'x_max': float(items[5]), 'y_max': float(items[6])}
        # calculate bbox3d corners
        obj3d = Object3d(l)
        bbox3d = obj3d.generate_corners3d()
        entry['bbox2d'] = bbox2d
        entry['bbox3d'] = bbox3d
        labels.append(entry)
    return labels



def cls_type_to_id(cls_type):
    type_to_id = {'forklift': 1, 'pallet': 2, 'agv': 3}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
