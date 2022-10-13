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

    # if (t := np.linalg.det(R)) < 0.0:
    #     print(t)
    #     R -= np.outer(U[:, 2], Vh[2, :] * 2.0)
    #     s[-1] *= -1.0

    M = np.identity(4)
    M[:3, :3] = R

    M[:3, 3] = art_mean
    T = np.identity(4)
    T[:3, 3] = -cam_mean

    M = np.dot(M, T)
    return M


def broadcast_transform(M):

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform = geometry_msgs.msg.TransformStamped()

    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "world"
    transform.child_frame_id = "zedm_base_link"

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


if __name__ == "__main__":
    rospy.init_node("world_to_zedm_transform")

    cam = np.array(
        [
            [0.4675, -0.12991, 0.076441],
            [0.4001, 0.11646, 0.034272],
            [0.37326, -0.17731, -0.052214],
        ]
    )

    art = np.array(
        [
            [-0.011744, 0.20327, -0.1815],
            [-0.013273, -0.043344, -0.17657],
            [0.14558, 0.21421, -0.17886],
        ]
    )
    # m = get_transformation()
    M = get_transformation(cam, art)

    broadcast_transform(M)
    # broadcast_transform(R, t)
