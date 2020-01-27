import os
from . import _transformation as _trafo_impl


# this is fairly hacky because it relies on the extension
# would be better to get this from z5 directly, but this is
# currently not exposed in the API
def get_path_and_key_from_dataset(dataset):
    extensions = ['.n5', '.zr', '.zarr']
    full_path = dataset.path
    path_fragments = full_path.split('/')
    file_index = [i for i, name in enumerate(path_fragments)
                  if os.path.splitext(name)[1].lower() in extensions]
    assert len(file_index) == 1
    file_index = file_index[0]
    path = '/'.join(path_fragments[:file_index+1])
    key = '/'.join(path_fragments[file_index+1:])
    return path, key


# TODO support pre-smoothing and support hdf5
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
