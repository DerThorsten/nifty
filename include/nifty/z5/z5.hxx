#pragma once

#include "z5/multiarray/xtensor_access.hxx"

namespace nifty{
namespace tools{

    template<class ARRAY, class COORD>
    inline void readSubarray(const z5::Dataset & ds,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             xt::xexpression<ARRAY> & subarray){
        z5::multiarray::readSubarray<typename ARRAY::value_type>(ds, subarray, beginCoord.begin());
    }

    template<class ARRAY, class COORD>
    inline void writeSubarray(z5::Dataset & ds,
                              const COORD & beginCoord,
                              const COORD & endCoord,
                              const xt::xexpression<ARRAY> & subarray){
        z5::multiarray::writeSubarray<typename ARRAY::value_type>(ds, subarray, beginCoord.begin());
    }

} // namespace nifty::tools
} // namespace nifty
