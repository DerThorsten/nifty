#pragma once

#include "z5/multiarray/xtensor_access.hxx"
#include "z5/dataset_factory.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/xtensor/xtensor.hxx"

#include "xtensor/xeval.hpp"

namespace nifty{
namespace nz5 {

    // TODO this should be moved to z5 itself
    // TODO make n-dimensional
    // we just support simple nearest upsampling
    template<class T>
    void nearestUpsampling(const std::string & inPath,
                           const std::vector<int> & samplingFactor,
                           const std::string & outPath,
                           const int numberOfThreads) {

        typedef T ValueType;
        typedef xt::xtensor<ValueType, 3> ArrayType;
        typedef typename ArrayType::shape_type ShapeType;
        std::vector<std::size_t> zeroCoord = {0, 0, 0};

        parallel::ThreadPool tp(numberOfThreads);

        auto dsIn = z5::openDataset(inPath);
        const auto & smallShape = dsIn->shape();
        const std::size_t nPix = std::accumulate(smallShape.begin(), smallShape.end(), 1, std::multiplies<std::size_t>());
        const std::vector<std::size_t> strides = {smallShape[2] * smallShape[1], smallShape[2], 1};

        // read all the input data
        ShapeType arrayShape = {smallShape[0], smallShape[1], smallShape[2]};
        ArrayType data(arrayShape);
        z5::multiarray::readSubarray<ValueType>(dsIn, data, zeroCoord.begin());

        auto dsOut = z5::openDataset(outPath);
        const auto & shape = dsOut->shape();
        const auto & chunkShape = dsOut->maxChunkShape();
        const std::size_t nChunks = dsOut->numberOfChunks();

        const auto & chunksPerDimension = dsOut->chunksPerDimension();
        const std::vector<std::size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        std::unordered_map<std::size_t, ArrayType> chunkStorage;
        std::vector<std::size_t> chunkProgress(nChunks, 0);

        std::mutex m;

        // std::cout << "Here !" << std::endl;
        parallel::parallel_foreach(tp, nPix, [&](const int tId, const std::size_t pixId){
            // find the coordinates
            std::size_t index = pixId;
            std::size_t posAtAxis;
            std::vector<std::size_t> coord(3);
            for(unsigned d = 0; d < 3; ++d) {
                posAtAxis = index / strides[d];
                index -= posAtAxis * strides[d];
                coord[d] = posAtAxis;
            }

            // std::cout << "AAA" << std::endl;
            const ValueType val = data(coord[0], coord[1], coord[2]);

            // upsample the coordinates
            std::vector<std::size_t> pixBegin(3), pixShape(3);
            for(unsigned d = 0; d < 3; ++d) {
                pixBegin[d] = coord[d] * samplingFactor[d];
                const std::size_t pixEnd = std::min((coord[d] + 1) * samplingFactor[d], shape[d]);
                pixShape[d] = pixEnd - pixBegin[d];
            }

            std::vector<std::vector<std::size_t>> chunkIds;
            const auto & chunking = dsOut->chunking();
            chunking.getBlocksOverlappingRoi(pixBegin, pixShape, chunkIds);

            // std::cout << "BBB" << std::endl;
            for(const auto & chunkId : chunkIds) {
                // std::cout << "CCC" << std::endl;
                // see if the chunk is already in the storage
                const std::size_t chunkIndex = chunkId[0] * chunkStrides[0] + chunkId[1] * chunkStrides[1] + chunkId[2] * chunkStrides[2];
                auto chunkIt = chunkStorage.begin();
                {
                    std::lock_guard<std::mutex> lock(m);
                    chunkIt = chunkStorage.find(chunkIndex);
                    // if it isn't, make the chunk array
                    if(chunkIt == chunkStorage.end()) {
                        std::vector<std::size_t> chunkShape;
                        dsOut->getChunkShape(chunkId, chunkShape);
                        ShapeType arrayShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
                        chunkIt = chunkStorage.emplace(chunkIndex, ArrayType(arrayShape)).first;
                    }
                }
                // std::cout << "CCC " << chunkIndex << std::endl;

                // find the local coordinates in the chunk and write the value and mask
                std::vector<std::size_t> offsetInRequest, shapeInRequest, offsetInChunk;
                // std::cout << "C1" << std::endl;
                chunking.getCoordinatesInRoi(chunkId, pixBegin, pixShape,
                                             offsetInRequest, shapeInRequest, offsetInChunk);
                // std::cout << offsetInChunk[0] << " " << offsetInChunk[1] << " " << offsetInChunk[2] << std::endl;
                // std::cout << shapeInRequest[0] << " " << shapeInRequest[1] << " " << shapeInRequest[2] << std::endl;
                // std::cout << "C2" << std::endl;
                auto & chunkArray = chunkIt->second;
                // std::cout << "C3" << std::endl;
                xt::xstrided_slice_vector offsetSlice;
                // std::cout << "C4" << std::endl;
                xtensor::sliceFromOffset(offsetSlice, offsetInChunk, shapeInRequest);
                // std::cout << "C5" << std::endl;
                auto view = xt::strided_view(chunkArray, offsetSlice);
                // std::cout << "C6" << std::endl;
                // std::cout << view.shape()[0] << " " << view.shape()[1] << " " << view.shape()[2] << std::endl;
                view = val;

                // std::cout << "DDD" << std::endl;
                // check if the chunk is complete and write out if it is
                const std::size_t requestSize = std::accumulate(shapeInRequest.begin(), shapeInRequest.end(), 1, std::multiplies<std::size_t>());
                const std::size_t totalSize = dsOut->getChunkSize(chunkId);
                bool chunkFinished = false;
                {
                    std::lock_guard<std::mutex> lock(m);
                    auto & progress = chunkProgress[chunkIndex];
                    progress += requestSize;
                    // this should never be bigger, but just to be sure...
                    if(progress >= totalSize) {
                        chunkFinished = true;
                    }
                }

                // std::cout << "EEE" << std::endl;
                if(chunkFinished) {
                    //don't write chunk if it is all zeros
                    auto chunkSum = xt::eval(xt::sum(chunkArray));
                    if(chunkSum(0) > 0) {
                        dsOut->writeChunk(chunkId, &chunkArray(0));
                    }
                }
            }
        });
    }


    inline void intersectMasks(const std::string & maskAPath,
                               const std::string & maskBPath,
                               const std::string & outPath,
                               const std::vector<std::size_t> & blockShape,
                               const int numberOfThreads) {
        typedef nifty::array::StaticArray<int64_t, 3> CoordType;
        typedef xt::xtensor<uint8_t, 3> ArrayType;
        typedef typename ArrayType::shape_type Shape;

        auto maskA = z5::openDataset(maskAPath);
        auto maskB = z5::openDataset(maskBPath);

        auto out = z5::openDataset(outPath);

        CoordType shape = {out->shape(0), out->shape(1), out->shape(2)};
        CoordType blocks = {blockShape[0], blockShape[1], blockShape[2]};
        parallel::ThreadPool tp(numberOfThreads);

        tools::parallelForEachBlock(tp, shape, blocks, [&](const int tid,
                                                           const CoordType & blockBegin,
                                                           const CoordType & blockEnd){
            // read masks
            Shape thisShape;
            for(unsigned d = 0; d < 3; ++d) {
                thisShape[d] = blockEnd[d] - blockBegin[d];
            }
            ArrayType arrayA(thisShape);
            ArrayType arrayB(thisShape);

            z5::multiarray::readSubarray<uint8_t>(maskA, arrayA, blockBegin.begin());
            z5::multiarray::readSubarray<uint8_t>(maskB, arrayB, blockBegin.begin());

            // intersect the mask 
            // TODO maskB should not be hard-coded to be inverted
            for(std::size_t z = 0; z < thisShape[0]; ++z) {
                for(std::size_t y = 0; y < thisShape[1]; ++y) {
                    for(std::size_t x = 0; x < thisShape[2]; ++x) {
                        arrayA(z, y, x) = arrayA(z, y, x) | (!arrayB(z, y, x));
                    }
                }
            }

            auto blockSum = xt::eval(xt::sum(arrayA));
            if(blockSum(0) > 0) {
                z5::multiarray::writeSubarray<uint8_t>(out, arrayA, blockBegin.begin());
            }
        });

        
    }


}
}
