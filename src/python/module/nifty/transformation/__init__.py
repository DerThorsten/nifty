from . import _transformation as _trafo_impl


def get_path_and_key_from_dataset(dataset):
    key = dataset.name
    # NOTE z5py will have this in newer version as well, but not in 2.0.3
    # which we currently depend on
    # h5py: datasets have a direct pointer to the parent file
    try:
        return dataset.file.filename, key
    except AttributeError:
        pass

    # go up the parents until we hit root (which is it's own parent)
    obj = dataset
    while obj.name != '/':
        obj = obj.parent
    return obj.filename, key


# TODO support pre-smoothing
def affineTransformationZ5(data, matrix, order, bounding_box,
                           fill_value=0, sigma=None):
    """
    """
    ndim = data.ndim
    start = [bb.start for bb in bounding_box]
    stop = [bb.stop for bb in bounding_box]
    if ndim != len(start) != len(stop):
        raise ValueError("Invalid dimension")
    dtype = str(data.dtype)
    func = "affineTransformationZ5%iD%s" % (ndim, dtype)
    func = getattr(_trafo_impl, func)
    path, key = get_path_and_key_from_dataset(data)
    return func(path, key, matrix, order, start, stop, fill_value)


# TODO support pre-smoothing
def affineTransformationH5(data, matrix, order, bounding_box,
                           fill_value=0, sigma=None):
    """
    """
    ndim = data.ndim
    start = [bb.start for bb in bounding_box]
    stop = [bb.stop for bb in bounding_box]
    if ndim != len(start) != len(stop):
        raise ValueError("Invalid dimension")
    dtype = str(data.dtype)
    func = "affineTransformationH5%iD%s" % (ndim, dtype)
    func = getattr(_trafo_impl, func)
    path, key = get_path_and_key_from_dataset(data)
    return func(path, key, matrix, order, start, stop, fill_value)


def affineTransformation(data, matrix, order, bounding_box,
                         fill_value=0):
    """
    """
    ndim = data.ndim
    start = [bb.start for bb in bounding_box]
    stop = [bb.stop for bb in bounding_box]
    if ndim != len(start) != len(stop):
        raise ValueError("Invalid dimension")
    dtype = str(data.dtype)
    func = "affineTransformation%iD%s" % (ndim, dtype)
    func = getattr(_trafo_impl, func)
    return func(data, matrix, order, start, stop, fill_value)
