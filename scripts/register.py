#!/usr/bin/env python3
import geometry_msgs.msg
import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_from_matrix


def get_transformation():
    """
    Computes the transormation from needed to map one triangle onto another. See:
    https://math.stackexchange.com/questions/158538/3d-transformation-two-triangles
    """

    # From zed-camera:
    # 1: 0.4675, -0.12991, 0.076441
    # 2: 0.4001, 0.11646, 0.034272
    # 3: 0.37326, -0.17731, -0.052214
    # 4: 0.30678, 0.06988, -0.09068

    # from ART
    # 1: -0.011744, 0.20327, -0.1815
    # 2: -0.013273, -0.043344, -0.17657
    # 3: 0.14558, 0.21421, -0.17886
    # 4: 0.14316, -0.044337, -0.18223

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

    c = (cam_1 + cam_2 + cam_3) / 3.0
    z = (art_1 + art_2 + art_3) / 3.0

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

    R = np.append(R, np.array([[0.0], [0.0], [0.0]]), axis=1)
    R = np.append(R, np.array([[0.0, 0.0, 0.0, 1.0]]), axis=0)

    print(R)

    return R, t


def get_identity_transformation():
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def broadcast_transform(R, t):
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform = geometry_msgs.msg.TransformStamped()

    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "world"
    transform.child_frame_id = "zedm_base_link"

    transform.transform.translation.x = t[0]
    transform.transform.translation.y = t[1]
    transform.transform.translation.z = t[2]

    quat = quaternion_from_matrix(R)
    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]

    broadcaster.sendTransform(transform)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("world_to_zedm_transform")
    R, t = get_transformation()
    broadcast_transform(R, t)
