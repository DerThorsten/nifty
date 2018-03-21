#pragma once

#include "z5/multiarray/xtensor_access.hxx"
#include "z5/dataset_factory.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace nz5 {

    // TODO make n-dimensional
    // we just support binary up / downsampling
    template<class T>
    void nearestUpsampling(const std::string & inPath,
                           const std::vector<int> & samplingFactor,
                           const std::string & outPath,
                           const int numberOfThreads) {

        typedef T ValueType;
        typedef xt::xtensor<ValueType, 3> ArrayType;
        typedef typename ArrayType::shape_type ShapeType;
        std::vector<size_t> zeroCoord = {0, 0, 0};

        parallel::ThreadPool tp(numberOfThreads);

        auto dsIn = z5::openDataset(inPath);
        const auto & smallShape = dsIn->shape();

        // read all the input data
        ArrayShape arrayShape(smallShape.begin(), smallShape.end());
        ArrayType data(arrayShape);
        z5::readSubarray<ValueType>(data, zeroCoord.begin());

        auto dsOut = z5::openDataset(outPath);
        const auto & shape = dsOut->shape();
        const auto & chunkShape = dsOut->maxChunkShape();
        const size_t nChunks = dsOut->numberOfChunks();

        const std::vector<size_t> strides = {smallShape[2] * smallShape[1], smallShape[2], 1};

        std::unordered_map<size_t, ArrayType> chunkStorage;
        std::vector<size_t> chunkProgress(nChunks, 0);
        std::vector<size_t> chunkSizes(nChunks);
        for(size_t chunkId = 0; chunkId < nChunks; ++chunkId) {
            chunkSizes[chunkId] = dsOut->getChunkSize(chunkId);
        }

        std::mutex m;

        parallel::parallel_foreach(tp, nPix, [&](const int tId, const size_t pixId){
            // find the coordinates
            size_t index = pixId;
            size_t posAtAxis;
            std::vector<size_t> coord(3);
            for(unsigned d = 0; d < DIM; ++d) {
                posAtAxis = index / strides[d];
                index -= posAtAxis * strides[d];
                coord[d] = posAtAxis;
            }

            const ValueType val = data(coord[0], coord[1], coord[2]);

            // upsample the coordinates
            std::vector<size_t> begin(3), end(3);
            for(unsigned d = 0; d < DIM; ++d) {
                begin[d] = coord[d] * samplingFactor[d];
                end[d] = std::min((coord[d] + 1) * samplingFactor[d], shape[d]);
            }

            // TODO implement, take the relevant params as argument
            std::vector<size_t> chunkIds;
            // std::unordered_map<size_t, >
            overlappingChunkIds(begin, end, chunkIds);
            for(const size_t chunkId : chunkIds) {
                // see if the chunk is already in the storage
                auto chunkIt = chunkStorage.begin();
                {
                    std::lock_guard lock(m);
                    chunkIt = chunkStorage.find(chunkId);
                    if(chunkIt == chunkStorage.end()) {
                        // TODO
                        // if it isn't, make the chunk array and
                    }
                }

                // find the local coordinates in the chunk and write the value and mask


                // check if the chunk is complete and write out if it is
            }
        });
    }


}
}
