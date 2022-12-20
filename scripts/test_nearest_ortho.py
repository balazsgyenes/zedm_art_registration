import numpy as np
import numpy.random as random

from algos import (random_affine_transform, random_corresponding_points,
    nearest_orthogonal_affine_transform)


if __name__ == "__main__":
    rng = random.default_rng()

    transform = random_affine_transform(rng)
    print(transform)

    world_coords, other_coords = random_corresponding_points(rng, transform, n_points=3)

    transform2 = nearest_orthogonal_affine_transform(other_coords, world_coords)

    print(transform2)
    assert np.allclose(transform, transform2)
