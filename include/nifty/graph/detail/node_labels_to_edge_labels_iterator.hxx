#pragma once
#ifndef NIFTY_GRAPH_DETAIL_NODE_LABELS_TO_EDGE_LABELS_ITERATOR_HXX
#define NIFTY_GRAPH_DETAIL_NODE_LABELS_TO_EDGE_LABELS_ITERATOR_HXX

#include <boost/iterator/transform_iterator.hpp>


namespace nifty{
namespace graph{
namespace detail_graph{

    template<class GRAPH, class NODE_MAP>
    class NodeLabelsToEdgeLabelsUnaryFunction{
    public:
        typedef const uint8_t & Reference;
        typedef uint8_t Value;
        typedef const uint8_t & reference;
        typedef uint8_t value;

        NodeLabelsToEdgeLabelsUnaryFunction(const GRAPH & graph, const NODE_MAP & nodeLabels)
        :   valBuffer_(),
            graph_(graph),
            nodeLabels_(nodeLabels){
        }


        NodeLabelsToEdgeLabelsUnaryFunction(const NodeLabelsToEdgeLabelsUnaryFunction & other)
        :   valBuffer_(other.valBuffer_),
            graph_(other.graph_),
            nodeLabels_(other.nodeLabels_){
        }

        const uint8_t & operator()(const int64_t edgeId) const {
            const auto uv = graph_.uv(edgeId);
            valBuffer_ =  nodeLabels_[uv.first] != nodeLabels_[uv.second] ? 1 : 0;
            return valBuffer_;
        }
    private:
        mutable uint8_t valBuffer_;
        const GRAPH & graph_;
        const NODE_MAP & nodeLabels_;
    };


    template<class GRAPH, class NODE_MAP>
    boost::transform_iterator<
        NodeLabelsToEdgeLabelsUnaryFunction<GRAPH, NODE_MAP> , typename GRAPH::EdgeIter, const uint8_t & , uint8_t
    > nodeLabelsToEdgeLabelsIterBegin(
        const GRAPH & graph,
        const NODE_MAP & nodeLabels
    ){
        typedef NodeLabelsToEdgeLabelsUnaryFunction<GRAPH, NODE_MAP> UFunc;
        typedef typename GRAPH::EdgeIter EdgeIter;
        UFunc uFunc(graph, nodeLabels);
        return boost::transform_iterator< UFunc , EdgeIter, const uint8_t & , uint8_t>(graph.edgesBegin(), uFunc);
    }




} // namespace detail_graph
} // namespace nifty::graph
} // namespace nifty


#endif /* NIFTY_GRAPH_DETAIL_NODE_LABELS_TO_EDGE_LABELS_ITERATOR_HXX */