// this implementation is inspired by: 
// https://github.com/bjoern-andres/seeded-region-growing/blob/master/include/andres/vision/seeded-region-growing.hxx


#pragma once
#ifndef NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX
#define NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"



namespace nifty {
namespace region_growing {




/// Seeded region growing in an n-dimension array, using the 2n-neighborhood
///
/// This function operates in-place on its second parameter.
///
/// \param elevation 8-bit Elevation map 
/// \param seeds As input: labeled seeds; as output: labeled grown regions
///
template<size_t DIM, class INTEGRAL_PIXEL_TYPE,  class T>
void 
seededRegionGrowing(
    const marray::View<INTEGRAL_PIXEL_TYPE> & elevation,
    marray::View<T>& seeds,
    const size_t numberOfQueues = 256
) {

    typedef array::StaticArray<int64_t, DIM> CoordType;
    CoordType shape;

    if(elevation.dimension() == DIM ) {
        throw std::runtime_error("dimension of elevation is wrong");
    }
    if(seeds.dimension() == DIM ) {
        throw std::runtime_error("dimension of seeds is wrong");
    }
    for(size_t j = 0; j < elevation.dimension(); ++j) {
        if(elevation.shape(j) != seeds.shape(j)) {
            throw std::runtime_error("shape of elevation and seeds mismatch.");
        }
        shape[j] = seeds.shape(j);
    }


    // define 256 queues, one for each gray level.
    std::vector<std::queue<CoordType> > queues(numberOfQueues);


    tools::forEachCoordinate(shape, [&](const CoordType & coordU){
        const auto lU = seeds(coordU.asStdArray());
        for(auto d=0; d<DIM; ++d){
            if(coordU[d] + 1 < shape[d]){
                CoordType coordV = coordU;
                ++coordV[d];
                const auto lV = seeds(coordV.asStdArray());

                if(lU != lV){
                    if(lU == 0 && lV != 0){
                        const auto eV = elevation(coordV.asStdArray());
                        queues[eV].push(coordV);
                    }
                    else if(lU != 0 && lV == 0){
                        const auto eU = elevation(coordU.asStdArray());
                        queues[eU].push(coordU);
                    }
                }
            }
        }

    });



    // grow
    INTEGRAL_PIXEL_TYPE grayLevel = 0;
    for(;;) {
        while(!queues[grayLevel].empty()) {
            // label pixel and remove from queue
            const auto coordU = queues[grayLevel].front();
            const auto sU = seeds(coordU.asStdArray());
            queues[grayLevel].pop();


            // add unlabeled neighbors to queues
            for(auto d=0; d<DIM; ++d){
                if(coordU[d] + 1 < shape[d]){
                    CoordType coordV = coordU;
                    ++coordV[d];
                    auto & sV = seeds(coordV.asStdArray());
                    if(sV == 0){
                        const auto eV = elevation(coordV.asStdArray());
                        const auto queueIndex = std::max(eV, grayLevel);
                        sV = sU;
                        queues[queueIndex].push(coordV);
                    }
                }
                if(coordU[d] - 1 >= 0 ){
                    CoordType coordV = coordU;
                    --coordV[d];
                    auto & sV = seeds(coordV.asStdArray());
                    if(sV == 0){
                        const auto eV = elevation(coordV.asStdArray());
                        const auto queueIndex = std::max(eV, grayLevel);
                        sV = sU;
                        queues[queueIndex].push(coordV);
                    }
                }
            }
        }
        if(grayLevel == numberOfQueues-1) {
            break;
        }
        else {
            queues[grayLevel] = std::queue<CoordType>(); // free memory
            ++grayLevel;
        }
    }
}

} // namespace region_growing
} // namespace nifty

#endif // #ifndef NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX