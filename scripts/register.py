#!/usr/bin/env python3
import rospy
from scipy.spatial.transform import Rotation
from numba import njit, prange
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
        self.lower_left = np.array([0.317, -0.177, -0.13])
        self.upper_right = np.array([0.512, 0.113, 0.088])


    def run(self):
        rospy.spin()
   

    def point_cloud_callback(self, data) -> None:
        time1 = time.time()
        pointcloud_array = utils.pointcloud2_to_xyz_array(data)

        # Crop the pointcloud to a box of interest
        cropped_pointcloud_array = utils.filter_xyz_array(pointcloud_array, self.lower_left, self.upper_right)

        # Extend xyz coordinates to homogeneous coordinates
        extended_pointcloud_array = np.c_[cropped_pointcloud_array, np.ones(cropped_pointcloud_array.shape[0])]
        
        T = get_transformation()
        # I = get_identity_transformation()
        transformed_pointcloud_array = transform_pointcloud_array(T, extended_pointcloud_array)

        transformed_pointcloud = utils.xyz_array_to_pointcloud2(transformed_pointcloud_array[:, :3])
        
        computation_time = (time.time()-time1)*1000.0
        print(f"Transformation function took {computation_time:07.3f} ms to transform {len(transformed_pointcloud_array)} points.")

        self.pointcloud_publisher.publish(transformed_pointcloud)



@njit(parallel=True) 
def transform_pointcloud_array(transformation, pointcloud_array):  
    for i in prange(pointcloud_array.shape[0]):
        pointcloud_array[i] = np.dot(transformation, pointcloud_array[i])

    return pointcloud_array


@njit(parallel=True) 
def get_transformation():
    """
    Computes the transormation from needed to map one triangle onto another. See:
    https://math.stackexchange.com/questions/158538/3d-transformation-two-triangles
    """

    #From zed-camera:
    #1: 0.4675, -0.12991, 0.076441
    #2: 0.4001, 0.11646, 0.034272
    #3: 0.37326, -0.17731, -0.052214
    #4: 0.30678, 0.06988, -0.09068

    # from ART
    #1: -0.011744, 0.20327, -0.1815
    #2: -0.013273, -0.043344, -0.17657
    #3: 0.14558, 0.21421, -0.17886
    #4: 0.14316, -0.044337, -0.18223


    # These points are selected in RVIZ, first select 
    # the numbered points on the marker "KIRURC Zed Mini -> ART"
    # in correct order from the Zed-Mini Camera point cloud.
    # Then use the ART pointing device to point at the same points.
    # Enter the values (x,y,z) into the cam_# (ZED-cam) and
    # art_# (ART pointing device) in order.

    # From zed-camera:
    cam_1 = np.array([0.4675, -0.12991, 0.076441])
    cam_2 = np.array([0.4001, 0.11646, 0.034272])
    cam_3 = np.array([0.37326, -0.17731, -0.052214])

    # From ART-Pointing-Device:
    art_1 = np.array([-0.011744, 0.20327, -0.1815])
    art_2 = np.array([-0.013273, -0.043344, -0.17657])
    art_3 = np.array([0.14558, 0.21421, -0.17886])

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
    U, s, Vh = np.linalg.svd(H)

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
