#pragma once


#include <algorithm>

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/array/arithmetic_array.hxx"

#include "nifty/xtensor/xtensor.hxx"

#ifdef WITH_Z5
#include "nifty/z5/z5.hxx"
#endif


namespace nifty{
namespace graph{


template<size_t DIM,
         class LABELS,
         class PIXEL_ARRAY,
         class NODE_MAP>
void projectScalarNodeDataToPixels(const GridRag<DIM, LABELS> & graph,
                                   NODE_MAP & nodeData,
                                   PIXEL_ARRAY & pixelData,
                                   const int numberOfThreads=-1){

    typedef array::StaticArray<int64_t, DIM> Coord;

    const auto & labels = graph.labels();
    const auto & shape = graph.shape();

    // if scalar
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = xtensor::read(labels, coord.asStdArray());
        xtensor::write(pixelData, coord.asStdArray(), nodeData[node]);
    });

}


template<size_t DIM,
         class LABELS,
         class PIXEL_ARRAY,
         class NODE_MAP>
void projectScalarNodeDataToPixelsOutOfCore(const GridRag<DIM, LABELS> & graph,
                                            NODE_MAP & nodeData,
                                            PIXEL_ARRAY & pixelData,
                                            array::StaticArray<int64_t, DIM> blockShape,
                                            const int numberOfThreads=-1){

    typedef array::StaticArray<int64_t, DIM> Coord;
    typedef typename LABELS::value_type LabelType;
    typedef typename NODE_MAP::value_type DataType;
    typedef typename xt::xtensor<LabelType, DIM>::shape_type ArrayShape;

    const auto & labels = graph.labels();
    const auto & shape = graph.shape();

    nifty::parallel::ThreadPool threadpool(numberOfThreads);
    struct PerThreadData{
        xt::xtensor<LabelType, DIM> blockLabels;
        xt::xtensor<DataType, DIM> blockData;
    };

    ArrayShape arrayShape;
    for(unsigned d = 0; d < DIM; ++d) {
        arrayShape[d] = blockShape[d];
    }

    const size_t nThreads = threadpool.nThreads();
    std::vector<PerThreadData> perThreadDataVec(nThreads);
    parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
        perThreadDataVec[i].blockLabels.resize(arrayShape);
        perThreadDataVec[i].blockData.resize(arrayShape);
    });

    tools::parallelForEachBlock(threadpool, shape, blockShape,
        [&](const int tid, const Coord & blockBegin, const Coord & blockEnd){

        const Coord actualBlockShape = blockEnd - blockBegin;
        ArrayShape blockArrayShape;
        for(unsigned d = 0; d < DIM; ++d) {
            blockArrayShape[d] = actualBlockShape[d];
        }

        auto & blockLabels = perThreadDataVec[tid].blockLabels;
        auto & blockData = perThreadDataVec[tid].blockData;

        bool haveLabelShape = true;
        for(unsigned d = 0; d < DIM; ++d) {
            if(actualBlockShape[d] != blockLabels.shape()[d]) {
                haveLabelShape = false;
                break;
            }
        }
        if(!haveLabelShape) {
            blockLabels.resize(blockArrayShape);
        }

        bool haveDataShape = true;
        for(unsigned d = 0; d < DIM; ++d) {
            if(actualBlockShape[d] != blockData.shape()[d]) {
                haveDataShape = false;
                break;
            }
        }
        if(!haveDataShape) {
            blockData.resize(blockArrayShape);
        }

        tools::readSubarray(labels, blockBegin, blockEnd, blockLabels);

        tools::forEachCoordinate(actualBlockShape, [&](const Coord & coord){
            const auto node = xtensor::read(blockLabels, coord.asStdArray());
            xtensor::write(blockData, coord.asStdArray(), nodeData[node]);
        });

        tools::writeSubarray(pixelData, blockBegin, blockEnd, blockData);
    });
}


} // namespace graph
} // namespace nifty
