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
    std::vector<std::pair<typename RAG::LabelType, typename RAG::LabelType>> & adjacencyOut,
    const int numberOfThreads=-1
) {
    typedef array::StaticArray<int64_t,3> Coord;
    typedef array::StaticArray<int64_t,2> Coord2;
    typedef typename RAG::LabelsProxy LabelsProxy;
    typedef typename LabelsProxy::BlockStorageType LabelStorage;

    typedef typename RAG::LabelType LabelType;
    typedef std::set<std::pair<LabelType, LabelType>> AdjacencySet;

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
    const Coord  sliceShape3({1L, shape[1], shape[2]});

    // instantiate the label storage
    LabelStorage labelsAStorage(threadpool, sliceShape3, nThreads);
    LabelStorage labelsBStorage(threadpool, sliceShape3, nThreads);

    size_t nSlices = shape[0];
    // iterate over all the slices z and find the adjacency to the slices above,
    // from z+2 to z+longRange
    parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int slice) {

        // get thread
        auto & threadAdjacency = threadData[tid];

        // get lower segmentation
        Coord beginA ({int64_t(slice), 0L, 0L});
        Coord endA({int64_t(slice + 1), shape[1], shape[2]});
        auto labelsA = labelsAStorage.getView(tid);
        labelsProxy.readSubarray(beginA, endA, labelsA);
        auto labelsASqueezed = labelsA.squeezedView();

        // get view for upper segmentation
        auto labelsB = labelsBStorage.getView(tid);

        for(int64_t z = 2; z <= longRange; ++z) {

            // get upper segmentation
            Coord beginB ({slice + z, 0L, 0L});
            Coord endB({slice + z + 1, shape[1], shape[2]});
            labelsProxy.readSubarray(beginB, endB, labelsB);
            auto labelsBSqueezed = labelsB.squeezedView();

            // iterate over the xy-coordinates
            LabelType lU, lV;
            tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
                lU = labelsASqueezed(coord.asStdArray());
                lV = labelsASqueezed(coord.asStdArray());
                threadAdjacency.insert(std::make_pair(std::min(lU, lV), std::max(lU, lV)));
            });
        }
    });

    // write the results to out vector
    // out size and thread offsets
    size_t totalSize = 0;
    std::vector<size_t> threadOffsets(nThreads);
    for(int tId = 0; tId < nThreads; ++tId) {
        auto threadSize = threadData[tId].size();
        threadOffsets[tId] = totalSize;
        totalSize += threadSize;
    }
    adjacencyOut.resize(totalSize);
    parallel::parallel_foreach(threadpool, nThreads, [&](const int tId, const int threadId){
        const auto & threadAdjacency = threadData[threadId];
        std::copy(threadAdjacency.begin(), threadAdjacency.end(), adjacencyOut.begin() + threadOffsets[threadId]);
    });
}


// TODO
// accumulate features for long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
template<class RAG, class AFFINITIES>
void accumulateLongRangeFeatures() {
    
}


} // end namespace graph
} // end namespace nifty
