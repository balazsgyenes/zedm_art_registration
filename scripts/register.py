#!/usr/bin/env python3
import rospy
import ros_numpy
from scipy.spatial.transform import Rotation
from numba import jit
from scipy import linalg


import numpy as np

from std_msgs.msg import String
import std_msgs.msg as std_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.msg as sensor_msgs
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2

class RegistrationManager:
    def __init__(self):
        rospy.init_node('zedm_art_registration_node', anonymous=True)

        self.art_listener = rospy.Subscriber("/art/body6/pose", PoseStamped, self.art_callback)
        self.pc_listener = rospy.Subscriber("/zedm/zed_node/point_cloud/cloud_registered", PointCloud2, self.point_cloud_callback)
        self.zed_camera_odometry_listener = rospy.Subscriber("/zedm/zed_node/odom", Odometry, self.odometry_callback)
        self.pc_publisher = rospy.Publisher('no_odometry_point_cloud', PointCloud2, queue_size=1)
        self.art_transformation = np.zeros((4,4))
        self.odometry_transformation = np.zeros((4,4))

    def run(self):
        rospy.spin()

    def to_inverse_affine_transformation_matrix(self, pos, rot):
        rotation_from_quat = Rotation.from_quat([rot.x, rot.y,rot.z,rot.w])
        rotation_matrix = rotation_from_quat.as_matrix()
        inv_rotation_matrix = linalg.inv(rotation_matrix)


        position = np.array([[pos.x], [pos.y],[pos.z]])
        inv_position = -position

        affine_append = np.array([[0,0,0,1]])

        affine_matrix = np.append(inv_rotation_matrix, inv_position, axis=1)
        affine_matrix = np.append(affine_matrix, affine_append, axis=0)

        return affine_matrix


    def art_callback(self, data):
        """From marker space to ART space"""
        rot = data.pose.orientation
        pos = data.pose.position
        self.art_transformation = self.to_inverse_affine_transformation_matrix(pos, rot)


    def odometry_callback(self, data)-> None:
        """From odometry (fake world) space to depth camera space.

        Removes the camera motion offset.
        """
        rot = data.pose.pose.orientation
        pos = data.pose.pose.position
        self.odometry_transformation = self.to_inverse_affine_transformation_matrix(pos, rot)

    def point_cloud_callback(self, data) -> None:
        # pointcloud2_to_xyz_array
        pc = ros_numpy.numpify(data)
        xyz_points = ros_numpy.point_cloud2.get_xyz_points(pc)
        xyzh_points = np.c_[xyz_points, np.ones(xyz_points.shape[0])]
        print(xyzh_points[0])

        #f = lambda x: np.dot(self.art_transformation, x)
        #transformed_pc = f(xyzh_points)

        transformed_pc = costly_fransform(self.art_transformation, xyzh_points)
        #transformed_pc = np.array([np.dot(self.art_transformation, x) for x in xyzh_points])
        #transformed_pc = xyzh_points
        # transformed_pc = np.expand_dims(transformed_pc, 0).astype('float')
        # xyzrgb_points = np.c_[xyzh_points, np.ones((xyzh_points.shape[0], 3))]
        print(transformed_pc.shape)
        new_pc = self.point_cloud(transformed_pc[:, :3])
        self.pc_publisher.publish(new_pc)


    def point_cloud(self, points, parent_frame="art"):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
    
        data = points.astype(dtype).tobytes()
    
        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
    
        header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
    
        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

@jit(nopython=True) 
def costly_fransform(transformation, point_array):
    for i in range(point_array.shape[0]):
        point_array[i] = np.dot(transformation, point_array[i])

    return point_array
    

if __name__ == '__main__':
    node = RegistrationManager()
    node.run()

