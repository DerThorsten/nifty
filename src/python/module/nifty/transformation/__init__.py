from . import _transformation as _trafo_impl


def get_path_and_key_from_dataset(dataset):
    key = dataset.name
    return dataset.file.filename, key


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


def coordinateTransformationZ5(data, coordinate_file, bounding_box, fill_value=0):
    """
    """
    ndim = data.ndim
    start = [bb.start for bb in bounding_box]
    stop = [bb.stop for bb in bounding_box]
    if ndim != len(start) != len(stop):
        raise ValueError("Invalid dimension")
    dtype = str(data.dtype)
    func = "coordinateTransformationZ5%iD%s" % (ndim, dtype)
    func = getattr(_trafo_impl, func)
    path, key = get_path_and_key_from_dataset(data)
    return func(path, key, coordinate_file, start, stop, fill_value)
