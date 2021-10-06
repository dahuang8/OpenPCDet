import open3d as o3d
import struct
import numpy as np


def convert_bin_to_pcd(bin_file):
    list_pcd = list()
    SIZE_FLOAT = 4
    with open(bin_file, "rb") as f:
        byte = f.read(SIZE_FLOAT * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(SIZE_FLOAT * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)


def load_kitti_pc(file_path):
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
