#pragma once


#include <algorithm>

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/array/arithmetic_array.hxx"

#include "nifty/xtensor/xtensor.hxx"


namespace nifty{
namespace graph{


template<size_t DIM,
         class LABELS_PROXY,
         class PIXEL_ARRAY,
         class NODE_MAP>
void projectScalarNodeDataToPixels(
    const GridRag<DIM, LABELS_PROXY> & graph,
    NODE_MAP & nodeData,
    PIXEL_ARRAY & pixelData,
    const int numberOfThreads = -1
){
    typedef array::StaticArray<int64_t, DIM> Coord;

    const auto labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto labels = labelsProxy.labels();

    // if scalar
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = xtensor::read(labels, coord.asStdArray());
        xtensor::write(pixelData, coord.asStdArray(), nodeData[node]);
    });

}


} // namespace graph
} // namespace nifty



