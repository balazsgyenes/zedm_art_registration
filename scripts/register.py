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
            [-0.28142, 0.12628, 1.5046],  # 1
            [0.0057246, 0.082931, 1.5142],  # 2
            [0.056722, -0.21053, 1.3361],  # 3
            [-0.26234, -0.23317, 1.2812],  # 4
            [-0.14176, -0.043698, 1.4159],  # 5
            [-0.23236, 0.11895, 1.2935],  # 6
            [-0.028546, 0.11938, 1.3199],  # 7
            [-0.020575, -0.087737, 1.1916],  # 8
            [-0.22111, -0.087544, 1.1664],  # 9
            [-0.098066, 0.019962, 1.2482],  # 10
            # [0.018275, -0.0063239, 1.2474],  # 11
            # [-0.13686, 0.014438, 1.2406],  # 12
        ]
    )

    panda = np.array(
        [
            [0.33141, -0.1974, 0.0599],  # 1
            [0.6213, -0.14786, 0.064271],  # 2
            [0.65211, 0.19953, 0.065273],  # 3
            [0.33042, 0.22866, 0.061007],  # 4
            [0.46341, 0.0034097, 0.061511],  # 5
            [0.35291, -0.076108, 0.2405],  # 6
            [0.55945, -0.07707, 0.24405],  # 7
            [0.55441, 0.16657, 0.24316],  # 8
            [0.35343, 0.16904, 0.23987],  # 9
            [0.48321, 0.040825, 0.24273],  # 10
            # [0.6005, 0.071865, 0.24356],  # 11
            # [0.44784, 0.048129, 0.24167],  # 12
        ]
    )
    # M = least_squares_fit_affine_transform(cam, panda)
    M = nearest_orthogonal_affine_transform(cam, panda)
    print(M)
    broadcast_transform(M)
    # broadcast_transform(R, t)
