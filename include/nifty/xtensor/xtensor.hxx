#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"

namespace nifty {

namespace xtensor {
    // small helper function to convert a ROI given by (offset, shape) into
    // a proper xtensor sliceing
    template<typename COORD>
    inline void sliceFromRoi(xt::slice_vector & roiSlice,
                             const COORD & begin,
                             const COORD & end) {
        for(int d = 0; d < offset.size(); ++d) {
            roiSlice.push_back(xt::range(begin[d], end[d]));
        }
    }
}

namespace tools {

    // TODO runtime checks
    template<class ARRAY1, class ARRAY2, class COORD>
    inline void readSubarray(const xt::xexpression<ARRAY1> & arrayExpression,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             xt::xexpression<ARRAY2> & subarrayExpression){
        auto & array = arrayExpression.derived_cast();
        auto & subarray = subarrayExpression.derived_cast();

        // get the view in the array
        xt::slice_vector slice(array);
        xtensor::sliceFromRoi(slice, beginCoord, endCoord);
        const auto view = xt::dynamic_view(array, slice);

        // FIXME this is probably slow and would be faster with direct memory copy ?!
        // or figure out xt assignments...
        subarray = view;
    }


    // TODO runtime checks
    template<class ARRAY1, class ARRAY2, class COORD>
    inline void writeSubarray(xt::xexpression<ARRAY1> & arrayExpression,
                              const COORD & beginCoord,
                              const COORD & endCoord,
                              const xt::xexpression<ARRAY2> & dataExpression){
        auto & array = arrayExpression.derived_cast();
        auto & data = dataExpression.derived_cast();

        // get the view in the array
        xt::slice_vector slice(array);
        xtensor::sliceFromRoi(slice, beginCoord, endCoord);
        auto view = xt::dynamic_view(array, slice);

        // FIXME this is probably slow and would be faster with direct memory copy ?!
        // or figure out xt assignments...
        view = data;
    }


} // namespace nifty::tools
} // namespace nifty
