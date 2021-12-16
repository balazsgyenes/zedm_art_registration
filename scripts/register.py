#!/usr/bin/env python3
import rospy
from scipy.spatial.transform import Rotation
from numba import jit, prange
from scipy import linalg
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import tf2_ros
import geometry_msgs.msg
import time
import utils

class RegistrationManager:
    def __init__(self):
        rospy.init_node('zedm_art_registration_node', anonymous=True)

        self.pointcloud_listener = rospy.Subscriber("/zedm/zed_node/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)
        self.pointcloud_publisher = rospy.Publisher('art_point_cloud', PointCloud2, queue_size=1)


    def run(self):
        rospy.spin()
   

    def point_cloud_callback(self, data) -> None:
        time1 = time.time()
        pointcloud_array = utils.pointcloud2_to_xyz_array(data)

        extended_pointcloud_array = np.c_[pointcloud_array, np.ones(pointcloud_array.shape[0])]
        
        T = get_transformation()
        I = get_identity_transformation()
        transformed_pointcloud_array = transform_pointcloud_array(T, extended_pointcloud_array)

        transformed_pointcloud = utils.xyz_array_to_pointcloud2(transformed_pointcloud_array[:, :3])
        
        print('Transformation function took %0.3f ms' % ((time.time()-time1)*1000.0))

        self.pointcloud_publisher.publish(transformed_pointcloud)



@jit(nopython=True, parallel=True) 
def transform_pointcloud_array(transformation, pointcloud_array):  
    for i in prange(pointcloud_array.shape[0]):
        pointcloud_array[i] = np.dot(transformation, pointcloud_array[i])

    return pointcloud_array


def get_transformation():
    """
    Computes the transormation from needed to map one triangle onto another. See:
    https://math.stackexchange.com/questions/158538/3d-transformation-two-triangles
    """

#From zed-camera:
#1: 0.38363, 0.11284, 0.015562
#2: 0.45257, -0.13175, 0.059208
#3: 0.29046, 0.064291, -0.11021
#4: 0.36034, -0.18044, -0.068268

#From ART-Pointing-Device:
#1: -0.19867, 0.076615, -0.23578
#2: -0.23415, 0.32177, -0.2342
#3: -0.043547, 0.10136, -0.23641
#4: -0.073239, 0.34946, -0.23872

    # These points are selected in RVIZ, first select 
    # the numbered points on the marker "KIRURC Zed Mini -> ART"
    # in correct order from the Zed-Mini Camera point cloud.
    # Then use the ART pointing device to point at the same points.
    # Enter the values (x,y,z) into the cam_# (ZED-cam) and
    # art_# (ART pointing device) in order.

    # From zed-camera:
    cam_1 = np.array([0.45257, -0.13175, 0.059208])
    cam_2 = np.array([0.38363, 0.11284, 0.015562])
    cam_3 = np.array([0.36034, -0.18044, -0.068268])

    # From ART-Pointing-Device:
    art_1 = np.array([-0.23415, 0.32177, -0.2342])
    art_2 = np.array([-0.19867, 0.076615, -0.23578])
    art_3 = np.array([-0.073239, 0.34946, -0.23872])

    c = (cam_1 + cam_2 + cam_3)/3.0
    z = (art_1 + art_2 + art_3)/3.0

    # Centered
    ycam_1 = cam_1 - c
    ycam_2 = cam_2 - c
    ycam_3 = cam_3 - c

    yart_1 = art_1 - z
    yart_2 = art_2 - z
    yart_3 = art_3 - z

    # Build outer products
    y1 = np.outer(ycam_1, yart_1)
    y2 = np.outer(ycam_2, yart_2)
    y3 = np.outer(ycam_3, yart_3)

    H = y1 + y2 + y3

    # Compute singular value decomposition
    U, s, Vh = linalg.svd(H)

    # Compute the rotation matrix
    R = Vh.T.dot(U.T)

    # Compute the translation
    t = z - R.dot(c)
    t = np.array([[t[0]], [t[1]], [t[2]]])
    
    affine_append = np.array([[0,0,0,1]])
    affine_matrix = np.append(R, t, axis=1)
    affine_matrix = np.append(affine_matrix, affine_append, axis=0)
    
    return affine_matrix

def get_identity_transformation():
    return np.array([
    [ 1.,         0.,         0.,          0.],
    [ 0.,         1.,         0.,          0.],
    [ 0.,         0.,         1.,          0.],
    [ 0.,         0.,         0.,          1.]])


if __name__ == '__main__':
    node = RegistrationManager()
    node.run()
