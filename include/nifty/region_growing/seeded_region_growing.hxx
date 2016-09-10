// THIS FILE IS A MODICTION OF 
// https://github.com/bjoern-andres/seeded-region-growing/blob/master/include/andres/vision/seeded-region-growing.hxx

// Seeded region growing in n-dimensional grid graphs, in linear time.
//
// Copyright (c) 2013 by Bjoern Andres.
// 
// This software was developed by Bjoern Andres.
// Enquiries shall be directed to bjoern@andres.sc.
//
// All advertising materials mentioning features or use of this software must
// display the following acknowledgement: ``This product includes andres::vision 
// developed by Bjoern Andres. Please direct enquiries concerning andres::vision 
// to bjoern@andres.sc''.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, 
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - All advertising materials mentioning features or use of this software must 
//   display the following acknowledgement: ``This product includes 
//   andres::vision developed by Bjoern Andres. Please direct enquiries 
//   concerning andres::vision to bjoern@andres.sc''.
// - The name of the author must not be used to endorse or promote products 
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
#pragma once
#ifndef NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX
#define NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "andres/vision/connected-components.hxx"


namespace nifty {
namespace region_growing {

namespace detail {
   template<class T> 
      inline bool isAtSeedBorder(const marray::View<T>& seeds, const size_t index);
}


/// Seeded region growing in an n-dimension array, using the 2n-neighborhood
///
/// This function operates in-place on its second parameter.
///
/// \param elevation 8-bit Elevation map 
/// \param seeds As input: labeled seeds; as output: labeled grown regions
///
template<class INTEGRAL_PIXEL_TYPE,  class T>
void 
seededRegionGrowing(
    const marray::View<INTEGRAL_PIXEL_TYPE> & elevation,
    marray::View<T>& seeds,
    const size_t numberOfQueues = 256
) {

    if(elevation.dimension() != seeds.dimension()) {
        throw std::runtime_error("dimension of elevation and seeds mismatch.");
    }
    for(size_t j = 0; j < elevation.dimension(); ++j) {
        if(elevation.shape(j) != seeds.shape(j)) {
            throw std::runtime_error("shape of elevation and seeds mismatch.");
        }
    }

    // define 256 queues, one for each gray level.
    std::vector<std::queue<size_t> > queues(numberOfQueues);

    // add each unlabeled pixels which is adjacent to a seed
    // to the queue corresponding to its gray level
    for(size_t j = 0; j < seeds.size(); ++j) {
        if(detail::isAtSeedBorder<T>(seeds, j)) {
            NIFTY_CHECK_OP(elevation(j),<,numberOfQueues, "wrong data in evaluation map\n");
            queues[elevation(j)].push(j);
        }
    }

    // grow
    INTEGRAL_PIXEL_TYPE grayLevel = 0;
    std::vector<size_t> coordinate(elevation.dimension());
    for(;;) {
        while(!queues[grayLevel].empty()) {
            // label pixel and remove from queue
            size_t j = queues[grayLevel].front();
            queues[grayLevel].pop();

            // add unlabeled neighbors to queues
            seeds.indexToCoordinates(j, coordinate.begin());
            for(auto d = 0; d < elevation.dimension(); ++d) {
                if(coordinate[d] != 0) {
                    --coordinate[d];
                    if(seeds(coordinate.begin()) == 0) {
                        size_t k;
                        seeds.coordinatesToIndex(coordinate.begin(), k);
                        INTEGRAL_PIXEL_TYPE queueIndex = std::max(elevation(coordinate.begin()), grayLevel);
                        seeds(k) = seeds(j); // label pixel
                        queues[queueIndex].push(k);
                    }
                    ++coordinate[d];
                }
            }
            for(auto d = 0; d < elevation.dimension(); ++d) {
                if(coordinate[d] < seeds.shape(d) - 1) {
                    ++coordinate[d];
                    if(seeds(coordinate.begin()) == 0) {
                        size_t k;
                        seeds.coordinatesToIndex(coordinate.begin(), k);
                        INTEGRAL_PIXEL_TYPE queueIndex = std::max(elevation(coordinate.begin()), grayLevel);
                        seeds(k) = seeds(j); // label pixel
                        queues[queueIndex].push(k);
                    }
                    --coordinate[d];
                }
            }
        }
        if(grayLevel == numberOfQueues-1) {
            break;
        }
        else {
            queues[grayLevel] = std::queue<size_t>(); // free memory
            ++grayLevel;
        }
    }
}

/// Seeded region growing in an n-dimension array, using the 2n-neighborhood
///
/// This function operates in-place on its second parameter.
///
/// \param elevation 8-bit Elevation map 
/// \param seeds As input: labeled seeds; as output: labeled grown regions
///
template<size_t DIM, class INTEGRAL_PIXEL_TYPE,  class T>
void 
seededRegionGrowing2(
    const marray::View<INTEGRAL_PIXEL_TYPE> & elevation,
    marray::View<T>& seeds,
    const size_t numberOfQueues = 256
) {

    typedef array::StaticArray<int64_t, DIM> CoordType;
    CoordType shape;

    if(elevation.dimension() != seeds.dimension()) {
        throw std::runtime_error("dimension of elevation and seeds mismatch.");
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

// \cond SUPPRESS_DOXYGEN
namespace detail {

template<class T>
inline bool isAtSeedBorder(
    const marray::View<T>& seeds,
    const size_t index
) {
    if(seeds(index) == 0) { 
        return false; // not a seed voxel
    }
    else {
        std::vector<size_t> coordinate(seeds.dimension());
        seeds.indexToCoordinates(index, coordinate.begin());
        for(unsigned char d = 0; d < seeds.dimension(); ++d) {
            if(coordinate[d] != 0) {
                --coordinate[d];
                if(seeds(coordinate.begin()) == 0) {
                    return true;
                }
                ++coordinate[d];
            }
        }
        for(unsigned char d = 0; d < seeds.dimension(); ++d) {
            if(coordinate[d] < seeds.shape(d) - 1) {
                ++coordinate[d];
                if(seeds(coordinate.begin()) == 0) {
                    return true;
                }
                --coordinate[d];
            }
        }
        return false;
    }
}

} // namespace detail
// \endcond 

} // namespace region_growing
} // namespace nifty

#endif // #ifndef NIFTY_REGION_GROWING_SEEDED_REGION_GROWING_HXX