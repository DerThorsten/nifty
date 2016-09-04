#pragma once
#ifndef NIFTY_GRAPH_SUBGRAPH_MASKS_SUBGRAPH_WITH_CUT_HXX
#define NIFTY_GRAPH_SUBGRAPH_MASKS_SUBGRAPH_WITH_CUT_HXX


namespace nifty{
namespace graph{
namespace subgraph_masks{


    template<class GRAPH, class NODE_LABELS>
    struct SubgraphWithCutFromNodeLabels {

        typedef GRAPH GraphType;
        typedef NODE_LABELS NodeLabelsType;

        SubgraphWithCutFromNodeLabels(
            const GraphType & graph,
            const NodeLabelsType & nodeLabels
        )
        :   graph_(graph),
            nodeLabels_(nodeLabels){
        }

        bool useNode(const uint64_t v) const{ 
            return true; 
        }
        bool useEdge(const uint64_t graphEdge)const{ 
            const auto uv = graph_.uv(graphEdge);
            return nodeLabels_[uv.first] == nodeLabels_[uv.second];
        }
        const GraphType & graph_;
        const NodeLabelsType & nodeLabels_;
    };

}
}
}

#endif // NIFTY_GRAPH_SUBGRAPH_MASKS_SUBGRAPH_WITH_CUT_HXX