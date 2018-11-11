import numpy

def lifted_offsets(r):
    edge_set = set()
    edge_list = []

    for x0 in range(-r,r+1):
        for x1 in range(-r,r+1):   
            offset = (x0, x1)
            if offset not in edge_set and (-x0,-x1) not in edge_set:
                edge_set.add(offset)    
                edge_list.append(offset)

    return edge_list



def make_offset_dict(offsets):
    n_offsets = offsets.shape[0]
    offset_dict = dict()
    for i in range(n_offsets):
        x,y = edge_list[i]
        key = int(x), int(y)
        assert key not in offset_dict
        offset_dict[key] = i
    return offset_dict


def merge_offsets(offsets_a, offsets_b):
    offsets = numpy.concatenate([offsets_a, offsets_b], axis=0)
    n_offsets = offsets.shape[0]
    offset_set = set()
    for i in range(n_offsets):
        x,y = edge_list[i]
        key = int(x), int(y)
        offset_set.add(key)

    offset_dict = dict()
    offsets = []
    i = 0
    for key in offset_set:
        offset_dict[key] = i
        offsets.append(key)
        i += 1

    return offsets, offset_dict


def render_loss_augmented_2d(weights_offsets, weights, loss_offsets, loss):
    
    shape = weights.shape[0:2]


    weight_offset_dict = make_offset_dict(weights_offsets)
    loss_offset_dict = make_offset_dict(loss_offsets)



    merged_offsets, merged_offset_dict = merge_offsets(weights_offsets, loss_offsets)

    loss_augmented_weights_shape = shape + (len(merged_offset_dict),)
    loss_augmented_weights = numpy.zeros_like(loss_augmented_weights)

    
    for key in merged_offsets.keys():
        mi = merged_offsets[key]

        if key in weight_offset_dict:
            wi = weight_offset_dict[key]
            loss_augmented_weights[:, :, mi] += weights[:, :, wi]

        if key in loss_offset_dict:
            li = loss_offset_dict[key]
            loss_augmented_weights[:, :, mi] -= loss[:, :, wi]

    return loss_augmented_weights, merged_offsets