#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"
//
#include "vigra/multi_array_chunked.hxx"


namespace nifty{
namespace graph{



template<
    size_t DIM, 
    class LABELS_TYPE, 
    class PIXEL_ARRAY, 
    class NODE_MAP
>
void projectScalarNodeDataToPixels(
    const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    PIXEL_ARRAY & pixelData,
    const int numberOfThreads = -1
){
    typedef std::array<int64_t, DIM> Coord;

    const auto labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto labels = labelsProxy.labels(); 

    // if scalar 
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = labels(coord);
        pixelData(coord) = nodeData[node];
    });

}


template<
    class LABELS_TYPE, 
    class NODE_MAP,
    class SCALAR_TYPE
>
void projectScalarNodeDataToPixels(
    const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    vigra::ChunkedArray<3,SCALAR_TYPE> & pixelData,
    const int numberOfThreads = -1
){
    typedef std::array<int64_t, 2> Coord;

    const auto labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto & labels = labelsProxy.labels(); 
        
    vigra::Shape3 slice_shape(1, shape[1], shape[2]);

    vigra::MultiArray<3,LABELS_TYPE> this_labels(slice_shape);

    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    for(size_t z = 0; z < shape[0]; z++) {
        
        vigra::MultiArray<3,SCALAR_TYPE> this_data(slice_shape);
        
        // checkout this slice
        vigra::Shape3 slice_begin(z, 0, 0);
        labels.checkoutSubarray(slice_begin, this_labels);

        nifty::parallel::ThreadPool threadpool(pOpt);
        nifty::tools::parallelForEachCoordinate(threadpool,std::array<int64_t,2>({(int64_t)shape[2],(int64_t)shape[1]}) ,
        [&](int tid, const Coord & coord){
            const auto x = coord[0];
            const auto y = coord[1];
            const auto node = this_labels(0,y,x);
            this_data(0,y,x) = nodeData[node];
        });
        pixelData.commitSubarray(slice_begin,this_data);
    }

}


} // namespace graph
} // namespace nifty


#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX */
