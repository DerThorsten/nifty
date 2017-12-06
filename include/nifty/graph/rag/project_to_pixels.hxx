#pragma once


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/array/arithmetic_array.hxx"

//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{



template<size_t DIM,
         class LABELS_TYPE,
         class PIXEL_ARRAY,
         class NODE_MAP>
void projectScalarNodeDataToPixels(
    const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
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
        const auto node = labels(coord.asStdArray());
        pixelData(coord.asStdArray()) = nodeData[node];
    });

}


} // namespace graph
} // namespace nifty



