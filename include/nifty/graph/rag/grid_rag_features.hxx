#pragma once

#include "nifty/graph/rag/grid_rag.hxx"
#include "xtensor/xarray.hpp"
#include "nifty/xtensor/xtensor.hxx"

#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{

    template<size_t DIM, class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(const GridRag<DIM, LABELS_PROXY> & graph,
                                 const xt::xexpression<LABELS> & dataExp,
                                 NODE_MAP & nodeMap,
                                 const bool ignoreBackground = false,
                                 const typename LABELS::value_type ignoreValue = 0){
        typedef std::array<int64_t, DIM> Coord;
        typedef typename LABELS::value_type LabelType;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels();
        const auto & data = dataExp.derived_cast();

        std::vector<std::unordered_map<LabelType, size_t>> overlaps(graph.numberOfNodes());

        nifty::tools::forEachCoordinate(shape, [&](const Coord & coord){
            const auto node = xtensor::access(labels, coord);
            const auto l = xtensor::access(data, coord);
            overlaps[node][l] += 1;
        });

        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol
            size_t maxOl = 0;
            LABELS maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    if(ignoreBackground && kv.first == ignoreValue) {
                        continue;
                    }
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }
    }


} // end namespace graph
} // end namespace nifty
