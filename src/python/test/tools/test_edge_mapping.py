import numpy as np
import nifty.tools as nt


def test_edge_mapping():
    uv_ids = np.array([
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 7],
        [3, 4],
        [3, 7],
        [4, 5]
    ])
    # old nodes: 0  1  2  3  4  5  6  7
    new_nodes = [0, 1, 0, 2, 1, 2, 3, 3]

    edge_values = [1] * len(uv_ids)

    edge_mapping = nt.EdgeMapping(len(uv_ids))
    edge_mapping.initializeMapping(uv_ids, new_nodes)

    new_uv_ids = edge_mapping.getNewUvIds()

    new_uv_ids_exp = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [2, 3]
    ]
    )

    assert (new_uv_ids == new_uv_ids_exp).all()
    print "Passed uv-ids"

    new_values = edge_mapping.mapEdgeValues(edge_values)
    print new_values
    # new_values_expected =

    # assert (new_values == new_values_expected).all()
    # print "Passed values"

    edge_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    print edge_mapping.getNewEdgeIds(edge_ids)


if __name__ == '__main__':
    test_edge_mapping()
