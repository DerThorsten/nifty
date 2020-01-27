#pragma once
#include "nifty/transformation/coordinate_transformation.hxx"
#include "nifty/tools/blocking.hxx"

#ifdef WITH_Z5
#include "nifty/z5/z5.hxx"
#endif

// TODO support hdf5 as well
#ifdef WITH_HDF5
#endif

namespace nifty {
namespace transformation {


    template<unsigned NDIM, class DATASET, class BLOCKING, class CACHE>
    inline double readFromChunk(const array::StaticArray<int64_t, NDIM> & coord,
                                const DATASET & input, const BLOCKING & blocking, CACHE & chunkCache) {
        typedef typename CACHE::mapped_type ArrayType;
        typedef typename ArrayType::shape_type ShapeType;

        const uint64_t blockId = blocking.coordinatesToBlockId(coord);
        const auto & block = blocking.getBlock(blockId);
        const auto & blockBegin = block.begin();
        //
        array::StaticArray<int64_t, NDIM> coordInChunk;
        for(unsigned d = 0; d < NDIM; ++d) {
            coordInChunk[d] = coord[d] - blockBegin[d];
        }

        // find the chunk in our cache
        auto chunkIt = chunkCache.find(blockId);
        // chunk is not in cache
        if(chunkIt == chunkCache.end()) {
            // allocate this chunk's data
            ShapeType chunkShape;
            const auto & blockEnd = block.end();
            for(unsigned d = 0; d < NDIM; ++d) {
                chunkShape[d] = blockEnd[d] - blockBegin[d];
            }
            chunkIt = chunkCache.emplace(blockId, ArrayType(chunkShape)).first;

            tools::readSubarray(input, blockBegin, blockEnd, chunkIt->second);
        }

        return xtensor::read(chunkIt->second, coordInChunk);
    }


    // TODO support pre-smoothing (need fast-filters)
    template<unsigned NDIM, class DATASET, class ARRAY,
             class COORD_TRAFO, class INTERPOLATOR>
    void coordinateTransformationChunked(const DATASET & input, ARRAY & output,
                                         COORD_TRAFO && trafo, INTERPOLATOR && interpolator,
                                         const array::StaticArray<int64_t, NDIM> & start,
                                         const array::StaticArray<int64_t, NDIM> & stop){
                                         // const std::vector<double> & sigma,
                                         // const std::vector<int64_t> & halo){
        typedef array::StaticArray<int64_t, NDIM> CoordType;
        typedef array::StaticArray<double, NDIM> FloatCoordType;
        typedef typename ARRAY::value_type ValueType;

        const auto & shape = input.shape();
        array::StaticArray<int64_t, NDIM> maxRange;
        for(unsigned d = 0; d < NDIM; ++d) {
            maxRange[d] = shape[d] - 1;
        }

        // get the chunks
        const auto & chunks = tools::getChunkShape(input);

        // make the blocking
        CoordType bBegin, bShape, bBlockShape;
        for(unsigned d = 0; d < NDIM; ++d) {
            bShape[d] = shape[d];
            bBlockShape[d] = chunks[d];
        }
        tools::Blocking<NDIM> blocking(bBegin, bShape, bBlockShape);

        // initialize the chunk cache
        // would be nice to use LRU cache or simialr!
        typedef xt::xtensor<ValueType, NDIM> BlockArrayType;
        std::unordered_map<uint64_t, BlockArrayType> chunkCache;

        CoordType normalizedOutCoord;
        FloatCoordType coord;
        std::vector<CoordType> coordList;
        std::vector<double> weightList;

        tools::forEachCoordinate(start, stop, [&](const CoordType & outCoord){
            // transform the coordinate
            trafo(outCoord, coord);

            // range check
            for(unsigned d = 0; d < NDIM; ++d){
                if(coord[d] >= maxRange[d] || coord[d] < 0) {
                    return;
                }
            }

            // interpolate the coordinate
            interpolator(coord, coordList, weightList);

            // iterate over the interpolated coords and compute the output value
            double val = 0.;
            for(unsigned i = 0; i < coordList.size(); ++i) {
                val += weightList[i] * readFromChunk<NDIM>(coordList[i], input, blocking, chunkCache);
            }

            for(unsigned d = 0; d < NDIM; ++d){
                normalizedOutCoord[d] = outCoord[d] - start[d];
            }
            xtensor::write(output, normalizedOutCoord, val);
        });

    }

}
}
