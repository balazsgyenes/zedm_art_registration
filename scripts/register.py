#!/usr/bin/env python3
import geometry_msgs.msg
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tf


def get_transformation(v0, v1):
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]

    t0 = np.mean(v0, axis=1)
    t1 = np.mean(v1, axis=1)

    v0 = v0 - t0.reshape(3, 1)
    v1 = v1 - t1.reshape(3, 1)

    H = np.einsum("ij,ik->jk", v0, v1)

    U, s, Vh = np.linalg.svd(H)
    R = np.dot(U, Vh)

    if np.linalg.det(R) < 0.0:
        R -= np.outer(U[:, 2], Vh[2, :] * 2.0)
        s[-1] *= -1.0
        print(np.linalg.det(R))

    M = np.identity(4)
    M[:3, :3] = R

    M[:3, 3] = t1
    T = np.identity(4)
    T[:3, 3] = -t0

    M = np.dot(M, T)
    return M


def broadcast_transform(M):

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform = geometry_msgs.msg.TransformStamped()

    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "world"
    transform.child_frame_id = "zedm_base_link"

    quat = tf.quaternion_from_matrix(M)

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
    art = np.array(
        [
            [0.24659, 0.19616, 0.39307],
            [0.35993, 0.21301, 0.39462],
            [0.4487, 0.028446, 0.38294],
        ]
    )

    cam = np.array(
        [
            [0.47996, -0.19691, -0.040598],
            [0.34798, -0.13264, -0.11247],
            [0.43435, 0.10304, -0.065814],
        ]
    )
    # m = get_transformation()
    M = get_transformation(art, cam)

    broadcast_transform(M)
    # broadcast_transform(R, t)
