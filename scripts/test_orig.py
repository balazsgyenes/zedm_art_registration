import numpy as np
import numpy.random as random

from register import get_transformation
from registration import random_affine_transform, random_corresponding_points


if __name__ == "__main__":
    rng = random.default_rng()

    transform = random_affine_transform(rng)
    print(transform)

    world_coords, other_coords = random_corresponding_points(rng, transform, n_points=3)

    transform2 = get_transformation(other_coords, world_coords)

    print(transform2)
    assert np.allclose(transform, transform2)
