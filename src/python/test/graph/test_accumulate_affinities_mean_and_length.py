import unittest

from nifty.graph import accumulate_affinities_mean_and_length
import numpy as np


class TestAffinityAccumulation(unittest.TestCase):
    IMAGE_SHAPE = (10, 10, 10)

    def test_accumulate_affinity_mean_and_length(self):
        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

        # Generate some random affinities:
        random_affinities = np.random.uniform(size=(len(offsets),) + self.IMAGE_SHAPE).astype('float32')
        random_labels = np.random.randint(25, size=self.IMAGE_SHAPE)

        mean, count = accumulate_affinities_mean_and_length(random_affinities, offsets, random_labels,
                                                                 offset_weights=[2, 4, 1, 5])

    def test_accumulate_affinity_mean_and_length_inside_cluster(self):
        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

        # Generate some random affinities:
        random_affinities = np.random.uniform(size=(len(offsets),) + self.IMAGE_SHAPE).astype('float32')
        random_labels = np.random.randint(25, size=self.IMAGE_SHAPE)

        mean, count = accumulate_affinities_mean_and_length(
            random_affinities, offsets, random_labels,
            offset_weights=[2, 4, 1, 5])


if __name__ == '__main__':
    unittest.main()
