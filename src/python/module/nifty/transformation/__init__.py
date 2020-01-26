from . import _transformation as _trafo_impl


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
