#!/usr/bin/env python3
import math

import geometry_msgs.msg
import numpy as np
import rospy
import tf2_ros


def get_transformation(cam, art):
    cam = np.array(cam, dtype=np.float64, copy=False)[:3]
    art = np.array(art, dtype=np.float64, copy=False)[:3]

    cam_mean = np.mean(cam, axis=0)
    art_mean = np.mean(art, axis=0)

    cam = cam - cam_mean
    art = art - art_mean

    H = np.einsum("ij,ik->jk", cam, art)

    U, s, Vh = np.linalg.svd(H)
    R = np.dot(Vh.T, U.T)

    if np.linalg.det(R) < 0.0:
        print(np.linalg.det(R))
        # R[:, 1] *= -1.0
        # R[:, 0] *= -1.0
        R[:, 2] *= -1.0
        # R[:, 1], R[:, 2] = R[:, 2].copy(), R[:, 1].copy()
        # R[1], R[2] = R[2].copy(), R[1].copy()
        print(np.linalg.det(R))

    M = np.identity(4)
    M[:3, :3] = R

    M[:3, 3] = art_mean
    T = np.identity(4)
    T[:3, 3] = -cam_mean

    M = np.dot(M, T)

    return M


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
    transform.child_frame_id = "zedm_left_camera_frame"

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
            [0.31358, -0.18482, 0.019107],
            [0.33351, 0.033511, -0.022393],
            # [0.25871, -0.028601, -0.030549],
            [0.40935, -0.040489, -0.13591],
        ]
    )

    panda = np.array(
        [
            [0.38616, 0.27147, 0.2776],
            [0.53905, 0.11011, 0.25714],
            # [0.54555, 0.20716, 0.27335],
            [0.48835, 0.14241, 0.11909],
        ]
    )
    M = get_transformation(cam, panda)
    print(M)
    broadcast_transform(M)
    # broadcast_transform(R, t)
