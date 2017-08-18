#pragma once

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty {
namespace graph {


// TODO
// get the long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
template<class RAG>
void getLongRangeAdjacency(
    const RAG & rag,
    const size_t longRange,
    std::vector<typename RAG::LabelType, typename RAG::LabelType> & adjacencyOut,
    const int numberOfThreads=-1
) {
    typedef array::StaticArray<int64_t,3> Coord;
    typedef array::StaticArray<int64_t,2> Coord2;

    typedef typename RAG::LabelType LabelType;
    typedef AdjacencySet std::set<std::pair<LabelType, LabelType>>;

    // instantiate threadpool and get the actual number of threads
    parallel::ThreadPool threadpool(numberOfThreads);
    auto nThreads = threadpool.nThreads();

    // instantiate thread data (= adjacency set for each thread)
    std::vector<AdjacencySet> threadData(nThreads);

    // labels proxy and shape
    const auto & labelsProxy = rag.labelsProxy();
    const auto & shape = labelsProxy.shape();

    // instantiate the slice shapes
    const Coord2 sliceShape2({shape[1], shape[2]});
    const Coord  sliceShape3({1L,shape[1], shape[2]});

    // instantiate the label storage
    LabelStorage labelsAStorage(threadpool, sliceShape3, nThreads);
    LabelStorage labelsBStorage(threadpool, sliceShape3, nThreads);

    size_t nSlices = shape[0];
    // iterate over all the slices z and find the adjacency to the slices above,
    // from z+2 to z+longRange
    parallel::parallel_foreach(nSlices, threadpool, [&](const int tid, const int slice) {
        threadAdjacency = threadData[tid];
        for(size_t z = 2; z <= longRange; ++z) {
            tools::forEachCoordinate(sliceShape2, [&](const Coord2 coordU){

            );
        }
    });
}


// TODO
// accumulate features for long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
void accumulateLongRangeFeatures() {
    
}


} // end namespace graph
} // end namespace nifty
