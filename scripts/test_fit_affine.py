import numpy as np
import numpy.random as random

from registration import (random_affine_transform, fit_affine_transform,
    random_corresponding_points)


if __name__ == "__main__":
    rng = random.default_rng()

    transform = random_affine_transform(rng)
    print(transform)

    world_coords, other_coords = random_corresponding_points(rng, transform, n_points=20)

    transform2 = fit_affine_transform(other_coords, world_coords)

    print(transform2)
    assert np.allclose(transform, transform2)
