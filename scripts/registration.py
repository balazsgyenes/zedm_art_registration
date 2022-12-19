from typing import Tuple

import numpy as np
import numpy.random as random


def fit_affine_transform(other_coords, world_coords):
    """Computes the best fit affine transform from the other coordinate system
    into the world coordinate system.

    References:
    https://math.stackexchange.com/questions/613530/understanding-an-affine-transformation/613804
    https://en.m.wikipedia.org/wiki/Scale-invariant_feature_transform#Model_verification_by_linear_least_squares 
    """
    transform = np.identity(4)

    # construct left side of equation by expressing other_coords in homogenous
    # coordinates
    A = np.append(other_coords, np.ones((other_coords.shape[0], 1)), axis=1)

    # right side of equation is just the matching coordinates in world space
    b = world_coords

    # solve for model parameters (affine transform) by inverting A and
    # multiplying by b
    solution = np.matmul(np.linalg.pinv(A), b)

    transform[:3] = solution.T
    return transform


def random_affine_transform(rng: random.Generator):

    euler_angles = np.pi * rng.random(size=(3,)) - np.pi / 2
    translation = 5 * rng.random(size=(3,)) - 2.5

    alpha, beta, gamma = euler_angles
    c1, c2, c3 = np.cos(euler_angles)
    s1, s2, s3 = np.sin(euler_angles)

    rotation = np.array([
        [c1*c3 - c2*s1*s3,  -c1*s3 - c2*c3*s1,  s1*s2],
        [c3*s1 + c1*c2*s3,  c1*c2*c3-s1*s3,     -c1*s2],
        [s2*s3,             c3*s2,              c2],
    ])

    assert np.allclose(np.linalg.det(rotation), 1)
    assert np.allclose(np.matmul(rotation, rotation.T), np.identity(3))
    assert np.allclose(alpha, np.arctan(rotation[0,2]/-rotation[1,2]))
    beta2 = np.arctan(np.sqrt(1 - rotation[2,2] ** 2)/rotation[2,2])
    assert np.allclose(beta, beta2) or np.allclose(beta, -beta2)
    assert np.allclose(gamma, np.arctan(rotation[2,0]/rotation[2,1]))

    transform = np.identity(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def random_corresponding_points(
    rng: random.Generator,
    transform: np.ndarray,
    n_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

    other_coords = np.ones(shape=(4, n_points))
    other_coords[:3, :n_points] = 10 * rng.random(size=(3, n_points)) - 5
    world_coords = np.matmul(transform, other_coords)

    world_coords = world_coords[:3].T
    other_coords = other_coords[:3].T

    return world_coords, other_coords
