#!/usr/bin/env python3
import math

import geometry_msgs.msg
import numpy as np
import rospy
import tf2_ros
from algos import (least_squares_fit_affine_transform,
                   nearest_orthogonal_affine_transform)


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def broadcast_transform(M):

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform = geometry_msgs.msg.TransformStamped()

    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "panda_link0"
    transform.child_frame_id = "cam_frame"

    quat = quaternion_from_matrix(M)

    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]

    t = M[:, -1]
    transform.transform.translation.x = t[0]
    transform.transform.translation.y = t[1]
    transform.transform.translation.z = t[2]
    transform.transform

    broadcaster.sendTransform(transform)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("world_to_zedm_transform")

    cam = np.array(
        [
            [0.001747, -0.050412, 1.4198],  # 1
            [-0.21066, 0.046029, 1.451],  # 2
            [-0.25285, -0.082485, 1.3667],  # 3
            # [-0.087686, -0.15671, 1.3412],  # 4
            [-0.1308, 0.14391, 1.5322],  # 5
            [-0.33677, 0.11442, 1.4877],  # 6
            # [-0.33488, 0.36026, 1.6446],  # 7
            [-0.33607, -0.18611, 1.2999],  # 8
            [0.018436, -0.23493, 1.3151],  # 9
            # [0.10631, 0.21223, 1.6082],  # 10
        ]
    )

    panda = np.array(
        [
            [0.61752, -0.022722, 0.072349],  # 1
            [0.40842, -0.13664, 0.071443],  # 2
            [0.37134, 0.01915, 0.069596],  # 3
            # [0.52641, 0.10549, 0.069],  # 4
            [0.48214, -0.25727, 0.062706],  # 5
            [0.27348, -0.22388, 0.064509],  # 6
            # [0.27147, -0.50161, 0.061842],  # 7
            [0.24772, 0.13682, 0.06017],  # 8
            [0.61575, 0.19313, 0.066029],  # 9
            # [0.69824, -0.32347, 0.070483],  # 10
        ]
    )
    # M = least_squares_fit_affine_transform(cam, panda)
    M = nearest_orthogonal_affine_transform(cam, panda)
    print(M)
    broadcast_transform(M)
    # broadcast_transform(R, t)
