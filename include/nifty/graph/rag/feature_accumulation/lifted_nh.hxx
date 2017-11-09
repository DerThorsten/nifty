#pragma once

#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/array_tools.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty {
namespace graph {

template<class RAG>
class LiftedNh : public UndirectedGraph<>{

// TODO
// - make ready for 2d stacked
// - out of core
// - switch to more general offset description
public:
    typedef RAG Rag;
    typedef UndirectedGraph<> BaseType;

    typedef array::StaticArray<int64_t, 4> Coord4;
    typedef array::StaticArray<int64_t, 3> Coord3;
    typedef array::StaticArray<int64_t, 2> Coord2;

    // compute lifted nh from affinities
    template<typename ITER>
    LiftedNh(
        const RAG & rag,
        ITER offsetsIterBegin,
        ITER offsetsIterEnd,
        const int numberOfThreads=-1
    ) : offsets_(offsetsIterBegin, offsetsIterEnd)
    {
        initLiftedNh(rag, numberOfThreads);
    }

    const std::vector<std::vector<int>> & offsets() const {return offsets_;}

private:
    void initLiftedNh(
        const Rag & labels, const int numberOfThreads);

    std::vector<std::vector<int>> offsets_;
};


// TODO use block storage mechanism to make out of core
template<class RAG>
void LiftedNh<RAG>::initLiftedNh(
    const RAG & rag, const int numberOfThreads
) {

    //typedef tools::BlockStorage<LabelType> LabelStorage;
    const auto & labels = rag.labelsProxy().labels();

    // set the number of nodes in the graph == number of labels
    BaseType::assign(rag.labelsProxy().numberOfLabels());
    Coord3 shape;
    for(size_t d = 0; d < 3; ++d) {
        shape[d] = labels.shape(d);
    }

    // TODO parallelize properly
    // threadpool and actual number of threads
    //nifty::parallel::ThreadPool threadpool(numberOfThreads);
    //const size_t nThreads = threadpool.nThreads();

    // number of links = number of channels * number of pixels
    size_t nLinks = offsets_.size() * labels.size();

    // FIXME super dirty hack to get the index to offsets translator from marray
    Coord4 affShape;
    affShape[0] = offsets_.size();
    for(size_t d = 0; d < 3; ++d) {
        affShape[d+1] = shape[d];
    }

    // FIXME skip init
    //marray::Marray<int8_t> fakeAffinities(marray::InitializationSkipping, affShape.begin(), affShape.end());
    marray::Marray<int8_t> fakeAffinities(affShape.begin(), affShape.end());

    //
    // iterate over the links and insert the corresponding uv pairs into the NH
    //
    Coord4 affCoord;
    Coord3 cU, cV;
    int channelId = 0;
    std::vector<int> offset;
    for(size_t linkId = 0; linkId < nLinks; ++linkId) {
        fakeAffinities.indexToCoordinates(linkId, affCoord.begin());
        offset = offsets_[channelId];

        bool outOfRange = false;
        for(size_t d = 0; d < 3; ++d) {
            cU[d] = affCoord[d+1];
            cV[d] = affCoord[d+1] + offset[d];
            // range check
            if(cV[d] >= shape[d] || cV[d] < 0) {
                outOfRange = true;
                break;
            }
        }
        if(outOfRange) {
            continue;
        }

        auto u = labels(cU.asStdArray());
        auto v = labels(cV.asStdArray());

        // only do stuff if the labels are different
        if(u != v) {

            // only add an edge to the lifted nh if it is not
            // in the local one
            if(rag.findEdge(u, v) != -1) {
                continue;
            }

            BaseType::insertEdge(
                std::min(u, v), std::max(u, v)
            );
        }

        // decrease the channel id back to zero if we go over the max
        ++channelId;
        if(channelId >= offsets_.size()) {
            channelId = 0;
        }
    }
}


}
}
