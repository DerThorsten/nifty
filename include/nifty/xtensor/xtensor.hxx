#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"

namespace nifty {

namespace xtensor {
    // helper function to convert a ROI given by (begin, end) into
    // a proper xtensor sliceing
    template<typename COORD1, typename COORD2>
    inline void sliceFromRoi(xt::slice_vector & roiSlice,
                             const COORD1 & begin,
                             const COORD2 & end) {
        for(int d = 0; d < begin.size(); ++d) {
            roiSlice.push_back(xt::range(begin[d], end[d]));
        }
    }


    // helper function to convert a ROI given by (offset, shape) into
    // a proper xtensor sliceing
    template<typename COORD1, typename COORD2>
    inline void sliceFromOffset(xt::slice_vector & roiSlice,
                                const COORD1 & offset,
                                const COORD2 & shape) {
        for(int d = 0; d < offset.size(); ++d) {
            roiSlice.push_back(xt::range(offset[d], offset[d] + shape[d]));
        }
    }


    // helper function to squeeze along given dimension
    template<class ARRAY>
    inline auto squeezedView(xt::xexpression<ARRAY> & arrayExp) {
        auto & array = arrayExp.derived_cast();
        auto & shape = array.shape();
        xt::slice_vector squeeze;
        for(const auto s : shape) {
            if(s == 1) {
                squeeze.push_back(1);
            } else {
                squeeze.push_back(xt::all());
            }
        }
        return xt::strided_view(array, squeeze);
    }


    // FIXME there should be a more elegant way to do this
    // FIXME we need a default way to do this for D > 6
    // maybe just accessing it with an iterator `begin` works ?!
    template<class ARRAY, class COORD>
    inline typename ARRAY::value_type read(const xt::xexpression<ARRAY> & arrayExp,
                                           const COORD & coord) {
        auto & array = arrayExp.derived_cast();
        switch(coord.size()) {
            case 1: return array(coord[0]);
            case 2: return array(coord[0], coord[1]);
            case 3: return array(coord[0], coord[1], coord[2]);
            case 4: return array(coord[0], coord[1], coord[2], coord[3]);
            case 5: return array(coord[0], coord[1], coord[2], coord[3], coord[4]);
            case 6: return array(coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]);
            default: throw std::runtime_error("xtensor access with coordinate is only supported up to 6D");
        }
    }

    // FIXME there should be a more elegant way to do this
    // FIXME we need a default way to do this for D > 6
    // maybe just accessing it with an iterator `begin` works ?!
    template<class ARRAY, class COORD>
    inline void write(xt::xexpression<ARRAY> & arrayExp,
                      const COORD & coord,
                      const typename ARRAY::value_type val) {
        auto & array = arrayExp.derived_cast();
        switch(coord.size()) {
            case 1: array(coord[0]) = val; break;
            case 2: array(coord[0], coord[1]) = val; break;
            case 3: array(coord[0], coord[1], coord[2]) = val; break;
            case 4: array(coord[0], coord[1], coord[2], coord[3]) = val; break;
            case 5: array(coord[0], coord[1], coord[2], coord[3], coord[4]) = val; break;
            case 6: array(coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]) = val; break;
            default: throw std::runtime_error("xtensor access with coordinate is only supported up to 6D");
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
        xt::slice_vector slice;
        xtensor::sliceFromRoi(slice, beginCoord, endCoord);
        const auto view = xt::strided_view(array, slice);

        // FIXME this is probably slow and would be faster with direct memory copy ?!
        // or figure out xt assignments...
        // subarray = view;
        std::copy(view.begin(), view.end(), subarray.begin());
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
        xt::slice_vector slice;
        xtensor::sliceFromRoi(slice, beginCoord, endCoord);
        auto view = xt::strided_view(array, slice);

        // FIXME this is probably slow and would be faster with direct memory copy ?!
        // or figure out xt assignments...
        // FIXME this fails with bad alloc
        // view = data;
        std::copy(data.begin(), data.end(), view.begin());
    }

    template<class ARRAY>
    inline bool isChunked(const xt::xexpression<ARRAY> & arr) {
        return false;
    }

    // dummy to be compatible with z5 interface
    template<class ARRAY>
    inline std::vector<size_t> getChunkShape(const xt::xexpression<ARRAY> & arr) {
        return std::vector<size_t>();
    }


} // namespace nifty::tools
} // namespace nifty
