#pragma once

#include "xtensor/xarray.hpp"
#include "z5/dataset.hxx"
#include "z5/multiarray/xt_access.hxx"
#include "nifty/tools/block_access.hxx"

namespace nifty{
namespace tools{

    template<class T, class COORD>
    inline void readSubarray(const z5::Dataset & ds,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             xt::xarray<T> & subarray){
        z5::multiarray::readSubarray<T>(ds, subarray, beginCoord.begin());
    }

    template<class T, class COORD>
    inline void writeSubarray(z5::Dataset & ds,
                              const COORD & beginCoord,
                              const COORD & endCoord,
                              const xt::xarray<T> & subarray){
        z5::multiarray::writeSubarray<T>(ds, subarray, beginCoord.begin());
    }

} // namespace nifty::tools
} // namespace nifty
