#pragma once

#define HAVE_CPP11_INITIALIZER_LISTS
#define HAVE_CPP11_STD_ARRAY

#include <andres/marray.hxx>
#include "nifty/tools/runtime_check.hxx"


namespace nifty{
namespace marray{
    using namespace andres;
}

namespace tools{

    template<class T, class COORD>
    inline void readSubarray(
        const marray::View<T> array,
        const COORD & beginCoord,
        const COORD & endCoord,
        marray::View<T> & subarray
    ){

        const auto dim = array.dimension();
        std::vector<int64_t> subShape(array.dimension());
        for(auto d = 0 ; d<dim; ++d){
            subShape[d] = beginCoord[d] - endCoord[d];
        }
        subarray = array.view(beginCoord.begin(), subShape.begin());
    }
    
    template<class T, class COORD>
    inline void writeSubarray(
        marray::View<T> array,
        const COORD & beginCoord,
        const COORD & endCoord,
        const marray::View<T> & data
    ){
        const auto dim = array.dimension();
        
        COORD subShape;
        for(auto d = 0 ; d<dim; ++d){
            subShape[d] = endCoord[d] - beginCoord[d];
        }
        for(int d = 0; d < dim; ++d )
            NIFTY_CHECK_OP(subShape[d],==,data.shape(d),"Shapes don't match!")
        auto subarray = array.view(beginCoord.begin(), subShape.begin());
        
        // for dim < 4 we can use forEachCoordinate (this only works if COORD is nifty::array::StaticArray)
        if(dim <= 4) {
            forEachCoordinate(subShape, [&](const COORD & coord){
                subarray(coord.asStdArray()) = data(coord.asStdArray());
            });
        }
        else { // otherwise use iterators
            auto itArray = subarray.begin();
            auto itData  = data.begin();
            for(; itArray != subarray.end(); ++itArray, ++itData)
                *itArray = *itData;
        }
    }
    
    // dummy function, because we don't lock for marrays
    template<class T, class COORD>
    inline void readSubarrayLocked(
        const marray::View<T> array,
        const COORD & beginCoord,
        const COORD & endCoord,
        marray::View<T> & subarray
    ){
        readSubarray(array,beginCoord,endCoord,subarray);
    }
    
    template<class T, class COORD>
    inline void writeSubarrayLocked(
        marray::View<T> array,
        const COORD & beginCoord,
        const COORD & endCoord,
        const marray::View<T> & data
    ){
        writeSubarray(array,beginCoord,endCoord,data);
    }
}

}

