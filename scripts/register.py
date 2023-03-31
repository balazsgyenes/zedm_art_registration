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
            [0.0056565, 0.043111, 1.488],  # 1
            [-0.044989, 0.047404, 1.4867],  # 2
            [-0.30049, 0.14731, 1.515],  # 3
            [-0.34505, 0.16956, 1.5246],  # 4
            [0.030178, -0.18466, 1.3487],  # 5
            [-0.0081902, -0.15513, 1.3634],  # 6
            [-0.23362, -0.060004, 1.3922],  # 7
            [-0.28253, -0.067406, 1.3827],  # 8
            [0.03368, 0.10184, 1.3174],  # 9
            [-0.14773, 0.1078, 1.2979],  # 10
            [0.018275, -0.0063239, 1.2474],  # 11
            [-0.13686, 0.014438, 1.2406],  # 12
        ]
    )

    panda = np.array(
        [
            [0.61744, -0.10055, 0.064562],  # 1
            [0.5679, -0.10581, 0.064086],  # 2
            [0.31599, -0.22251, 0.061423],  # 3
            [0.27272, -0.24889, 0.0601],  # 4
            [0.63042, 0.16979, 0.06529],  # 5
            [0.59325, 0.13491, 0.064267],  # 6
            [0.37399, 0.023476, 0.060984],  # 7
            [0.32388, 0.033635, 0.060222],  # 8
            [0.62287, -0.056563, 0.244],  # 9
            [0.44226, -0.062681, 0.24145],  # 10
            [0.6005, 0.071865, 0.24356],  # 11
            [0.44784, 0.048129, 0.24167],  # 12
        ]
    )
    # M = least_squares_fit_affine_transform(cam, panda)
    M = nearest_orthogonal_affine_transform(cam, panda)
    print(M)
    broadcast_transform(M)
    # broadcast_transform(R, t)
