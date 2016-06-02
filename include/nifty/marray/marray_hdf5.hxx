#pragma once
#ifndef NIFTY_MARRAY_MARRAY_HDF5_HXX
#define NIFTY_MARRAY_MARRAY_HDF5_HXX

#include <nifty/marray/marray.hxx>

#define HAVE_CPP11_INITIALIZER_LISTS
#include <andres/marray.hxx>
#include <andres/marray-hdf5.hxx>

namespace nifty{
namespace marray{
    using namespace andres;
}
}

namespace nifty{
namespace marray{

   namespace hdf5 = andres::hdf5;
}
}

#endif
