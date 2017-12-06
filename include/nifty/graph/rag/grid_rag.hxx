#pragma once

#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>

//#include <parallel/algorithm>
#include <unordered_set>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/array_tools.hxx"

#include "nifty/graph/rag/grid_rag_labels_proxy.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag.hxx"
// TODO switch to xtensor
#include "nifty/marray/marray.hxx"


namespace nifty{
namespace graph{


template<class COORD>
COORD makeCoord2(const COORD & coord,const size_t axis){
    COORD coord2 = coord;
    coord2[axis] += 1;
    return coord2;
};


template<class LABELS_PROXY>
struct RefHelper{
    typedef const LABELS_PROXY & type;
};

template<size_t DIM, class LABEL_TYPE>
struct RefHelper<ExplicitLabels<DIM, LABEL_TYPE>>{
    typedef ExplicitLabels<DIM, LABEL_TYPE> type;
};



template<size_t DIM, class LABELS_PROXY>
class GridRag : public UndirectedGraph<>{
public:
    struct DontComputeRag{};
    typedef LABELS_PROXY LabelsProxy;

    struct SettingsType{
        SettingsType()
        :   numberOfThreads(-1),
            blockShape()
        {
            for(auto d=0; d<DIM; ++d)
                blockShape[d] = 100;
        }
        int numberOfThreads;
        array::StaticArray<int64_t, DIM> blockShape;
    };

    typedef GridRag<DIM, LABELS_PROXY> SelfType;
    typedef array::StaticArray<int64_t, DIM> ShapeType;

    friend class detail_rag::ComputeRag< SelfType >;


    GridRag(const LabelsProxy & labelsProxy, const SettingsType & settings = SettingsType())
    :   settings_(settings),
        labelsProxy_(labelsProxy)
    {
        detail_rag::ComputeRag<SelfType>::computeRag(*this, settings_);
    }

    template<class ITER>
    GridRag(const LabelsProxy & labelsProxy,
            ITER serializationBegin,
            const SettingsType & settings = SettingsType())
    :   settings_(settings),
        labelsProxy_(labelsProxy) {
        this->deserialize(serializationBegin);
    }

    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }

    const ShapeType & shape()const{
        return labelsProxy_.shape();
    }

    UndirectedGraph<> extractSubgraphFromRoi(const ShapeType & begin,
                                             const ShapeType & end,
                                             std::vector<int64_t> & innerEdgesOut) const {
        typedef typename LABELS_PROXY::LabelType LabelType;
        innerEdgesOut.clear();

        ShapeType subShape;
        for(int d = 0; d < DIM; ++d)
            subShape[d] = end[d] - begin[d];
        marray::Marray<LabelType> subLabels(subShape.begin(), subShape.end());

        tools::readSubarray(labelsProxy_, begin, end, subLabels);

        std::vector<LabelType> uniqueNodes;
        tools::uniques(subLabels, uniqueNodes);

        std::map<LabelType, LabelType> globalToLocalNodes;
        for(int ii = 0; ii < uniqueNodes.size(); ++ii) {
            globalToLocalNodes[uniqueNodes[ii]] = ii;
        }
        UndirectedGraph<> subGraph(uniqueNodes.size());

        auto makeCoord2 = [](const ShapeType & coord, const int d) {
            ShapeType coord2 = coord;
            coord2[d] += 1;
            return coord2;
        };

        // extract the inner uv ids
        // TODO parallelize
        std::set<std::pair<int64_t, int64_t>> innerEdges;
        tools::forEachCoordinate(subShape, [&](const ShapeType & coord){
            const auto lU = subLabels(coord.asStdArray());
            for(int d = 0; d < DIM; ++d) {
                if(coord[d] < subShape[d] - 1) {
                    auto coordV = makeCoord2(coord, d);
                    const auto lV = subLabels(coordV.asStdArray());
                    if(lU != lV) {
                        innerEdges.emplace( std::make_pair(std::min(lU, lV), std::max(lU, lV)) );
                    }
                }
            }
        });

        // convert the inner uv ids to global edge ids and construct new graph in local coordinates
        innerEdgesOut.reserve(innerEdges.size());
        for(auto it = innerEdges.begin(); it != innerEdges.end(); ++it) {
            const auto lU = it->first;
            const auto lV = it->second;
            innerEdgesOut.push_back(this->findEdge(lU, lV));

            subGraph.insertEdge(globalToLocalNodes[lU], globalToLocalNodes[lV]);
        }
        return subGraph;
    }

protected:
    GridRag(const LabelsProxy & labelsProxy, const SettingsType & settings, const DontComputeRag)
    :   settings_(settings),
        labelsProxy_(labelsProxy){

    }
protected:
    typedef typename RefHelper<LABELS_PROXY>::type StorageType;
    SettingsType settings_;
    StorageType labelsProxy_;
};


// TODO switch to xtensor
template<std::size_t DIM, class LABELS_TYPE>
using ExplicitLabelsGridRag = GridRag<DIM, ExplicitLabels<DIM, LABELS_TYPE>>;

} // end namespace graph
} // end namespace nifty
