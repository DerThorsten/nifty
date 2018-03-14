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
    const auto & labels = rag.labels();

    // set the number of nodes in the graph == number of labels
    auto numberOfLabels = rag.numberOfLabels();
    BaseType::assign(numberOfLabels);
    Coord3 shape;
    Coord4 affShape;
    affShape[0] = offsets_.size();
    for(int d = 0; d < 3; ++d) {
        shape[d] = labels.shape()[d];
        affShape[d+1] = labels.shape()[d];
    }

    // TODO parallelize properly
    // threadpool and actual number of threads
    nifty::parallel::ThreadPool threadpool(numberOfThreads);
    const size_t nThreads = threadpool.nThreads();

    // per thread data for adjacencies
    struct PerThread{
        std::vector< container::BoostFlatSet<uint64_t> > adjacency;
    };
    std::vector<PerThread> threadAdjacencies(nThreads);
    parallel::parallel_foreach(threadpool, nThreads, [&](int tid, int threadId) {
        threadAdjacencies[threadId].adjacency.resize(numberOfLabels);
    });

    tools::parallelForEachCoordinate(threadpool, affShape, [&](int tid, const Coord4 & affCoord) {

        Coord3 cU, cV;
        const auto & offset = offsets_[affCoord[0]];

        for(int d = 0; d < 3; ++d) {
            cU[d] = affCoord[d+1];
            cV[d] = affCoord[d+1] + offset[d];
            // range check
            if(cV[d] < 0 || cV[d] >= shape[d]) {
                return;
            }
        }

        const auto u = xtensor::read(labels, cU.asStdArray());
        const auto v = xtensor::read(labels, cV.asStdArray());

        // only do stuff if the labels are different
        if(u != v) {

            // only add an edge to the lifted nh if it is not
            // in the local one
            if(rag.findEdge(u, v) != -1) {
                return;
            }
            auto & adjacency = threadAdjacencies[tid].adjacency;
            adjacency[v].insert(u);
            adjacency[u].insert(v);
        }
    });

    BaseType::mergeAdjacencies(threadAdjacencies, threadpool);
}


}
}
