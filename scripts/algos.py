from typing import Tuple

import numpy as np
import numpy.random as random


def nearest_orthogonal_affine_transform(
    other_coords: np.ndarray,
    world_coords: np.ndarray,
) -> np.ndarray:
    """Fits an affine transform using Singular Value Decomposition to find the
    nearest orthogonal matrix to `A^T b` (see link below). The returned
    transform converts points from other_coords to points in world_coords.

    https://en.wikipedia.org/wiki/Singular_value_decomposition#Nearest_orthogonal_matrix
    
    When given only 3 pairs of points, this algorithm works only roughly half
    the time, outputting incorrect results the other half. This case is
    detectable because the determinant of the rotation matrix is also negative,
    but I do not know how to recover from this. Retrying the algorithm with
    the points reordered seems to work. For 4 points, this failure case is much
    less common, and appears to be extremely unlikely for even more points.
    
    The normalization step is required, output is incorrect without it.
    """
    
    other_coords_mean = np.mean(other_coords, axis=0)
    world_mean = np.mean(world_coords, axis=0)

    other_coords = other_coords - other_coords_mean
    world_coords = world_coords - world_mean

    H = np.matmul(other_coords.T, world_coords)

    U, s, Vh = np.linalg.svd(H)

    # this is almost the pseudoinverse of H, but without the s
    R = Vh.T @ U.T

    if np.linalg.det(R) < 0.0:
        print(R)
        raise RuntimeError("Negative determinant of rotation matrix")

    M = np.identity(4)
    M[:3, :3] = R
    M[:3, 3] = world_mean

    T = np.identity(4)
    T[:3, 3] = -other_coords_mean
    M = M @ T

    return M


def least_squares_fit_affine_transform(
    other_coords: np.ndarray,
    world_coords: np.ndarray,
) -> np.ndarray:
    """Fits an affine transform using least-squares fitting given pairs of
    matching points. The returned transform converts points from other_coords
    to points in world_coords. This algorithm requires minimum 4 pairs of
    points.

    References:
    https://math.stackexchange.com/questions/613530/understanding-an-affine-transformation/613804
    https://en.m.wikipedia.org/wiki/Scale-invariant_feature_transform#Model_verification_by_linear_least_squares 
    """
    if other_coords.shape[0] < 4 or world_coords.shape[0] < 4:
        raise ValueError("Least squares fit requires at least 4 points.")

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


def random_affine_transform(rng: random.Generator) -> np.ndarray:

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
