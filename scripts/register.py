#!/usr/bin/env python3
import rospy
from scipy.spatial.transform import Rotation
from numba import jit
from scipy import linalg


import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2

# from std_msgs.msg import String
# import std_msgs.msg as std_msgs
# import sensor_msgs.msg as sensor_msgs
# from nav_msgs.msg import Odometry
# import sensor_msgs.point_cloud2 as pc2


import utils

class RegistrationManager:
    def __init__(self):
        rospy.init_node('zedm_art_registration_node', anonymous=True)

        self.art_listener = rospy.Subscriber("/art/body6/pose", PoseStamped, self.art_callback)
        self.pointcloud_listener = rospy.Subscriber("/zedm/zed_node/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)
        self.pointcloud_publisher = rospy.Publisher('no_odometry_point_cloud', PointCloud2, queue_size=1)
        self.art_transformation = np.zeros((4,4))


        #self.zed_camera_odometry_listener = rospy.Subscriber("/zedm/zed_node/odom", Odometry, self.odometry_callback)
        #self.odometry_transformation = np.zeros((4,4))

    def run(self):
        rospy.spin()



    def art_callback(self, data):
        """From marker space to ART space"""
        rot = data.pose.orientation
        pos = data.pose.position
        self.art_transformation = to_inverse_affine_transformation_matrix(pos, rot)


    # def odometry_callback(self, data)-> None:
    #     """From odometry (fake world) space to depth camera space.
    #
    #     Removes the camera motion offset.
    #     """
    #     rot = data.pose.pose.orientation
    #     pos = data.pose.pose.position
    #     self.odometry_transformation = self.to_inverse_affine_transformation_matrix(pos, rot)

    def point_cloud_callback(self, data) -> None:
        pointcloud_array = utils.pointcloud2_to_xyz_array(data)
        extended_pointcloud_array = np.c_[pointcloud_array, np.ones(pointcloud_array.shape[0])]
        transformed_pointcloud_array = transform_pointcloud_array(self.art_transformation, extended_pointcloud_array)

        transformed_pointcloud = utils.xyz_array_to_pointcloud2(transformed_pointcloud_array[:, :3])
        print(transformed_pointcloud_array.shape)
        self.pointcloud_publisher.publish(transformed_pointcloud)



@jit(nopython=True, parallel=True) 
def transform_pointcloud_array(transformation, pointcloud_array):
    for i in range(pointcloud_array.shape[0]):
        pointcloud_array[i] = np.dot(transformation, pointcloud_array[i])

    return pointcloud_array

def to_inverse_affine_transformation_matrix(pos, rot):
    rotation_from_quat = Rotation.from_quat([rot.x, rot.y,rot.z,rot.w])
    rotation_matrix = rotation_from_quat.as_matrix()
    inv_rotation_matrix = linalg.inv(rotation_matrix)


    position = np.array([[pos.x], [pos.y],[pos.z]])
    inv_position = -position

    affine_append = np.array([[0,0,0,1]])

    affine_matrix = np.append(inv_rotation_matrix, inv_position, axis=1)
    affine_matrix = np.append(affine_matrix, affine_append, axis=0)

    return affine_matrix


if __name__ == '__main__':
    node = RegistrationManager()
    node.run()

