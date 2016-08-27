#pragma once
#ifndef NIFTY_GRAPH_DETAIL_CONTIGUOUS_INDICES_HXX
#define NIFTY_GRAPH_DETAIL_CONTIGUOUS_INDICES_HXX


#include "nifty/graph/graph_tags.hxx"

namespace nifty{
namespace graph{
namespace detail_graph{

    template<class GRAPH, class EDGE_ID_TAG>
    class EdgeIndicesToContiguousEdgeIndicesImpl{
    public:
        typedef GRAPH Graph;
        EdgeIndicesToContiguousEdgeIndicesImpl(const Graph & graph)
        : ids_(graph){

            auto cid = 0;
            for(const auto edge : graph.edges()){
                ids_[edge] = cid;
                ++cid;
            }        

        }
        int64_t operator[](const int64_t edge)const{
            return ids_[edge];
        }
    private:
        typedef typename Graph:: template EdgeMap<uint64_t>  ContiguousIdStorage;
        ContiguousIdStorage ids_;
    };


    template<class GRAPH>
    class EdgeIndicesToContiguousEdgeIndicesImpl<GRAPH, nifty::graph::ContiguousTag>  {
    public:
        typedef GRAPH Graph;
        EdgeIndicesToContiguousEdgeIndicesImpl(const Graph & graph){
        }
        int64_t operator[](const int64_t edge)const{
            return edge;
        }
    private:
    };








    template<class GRAPH>
    class EdgeIndicesToContiguousEdgeIndices 
    :   public  EdgeIndicesToContiguousEdgeIndicesImpl<GRAPH, typename GRAPH::EdgeIdTag>{
    public:
        using EdgeIndicesToContiguousEdgeIndicesImpl<GRAPH, typename GRAPH::EdgeIdTag>::EdgeIndicesToContiguousEdgeIndicesImpl;
    };


    template<class GRAPH, class NODE_ID_TAG>
    class NodeIndicesToContiguousNodeIndicesImpl{
    public:
        typedef GRAPH Graph;
        NodeIndicesToContiguousNodeIndicesImpl(const Graph & graph)
        : ids_(graph){

            auto cid = 0;
            for(const auto node : graph.nodes()){
                ids_[node] = cid;
                ++cid;
            }        

        }
        int64_t operator[](const int64_t node)const{
            return ids_[node];
        }
    private:
        typedef typename Graph:: template NodeMap<uint64_t>  ContiguousIdStorage;
        ContiguousIdStorage ids_;
    };


    template<class GRAPH>
    class NodeIndicesToContiguousNodeIndicesImpl<GRAPH, nifty::graph::ContiguousTag>  {
    public:
        typedef GRAPH Graph;
        NodeIndicesToContiguousNodeIndicesImpl(const Graph & graph){
        }
        int64_t operator[](const int64_t node)const{
            return node;
        }
    private:
    };


    

    template<class GRAPH>
    class NodeIndicesToContiguousNodeIndices 
    :   public  NodeIndicesToContiguousNodeIndicesImpl<GRAPH, typename GRAPH::EdgeIdTag>{
    public:
        using EdgeIndicesToContiguousEdgeIndicesImpl<GRAPH, typename GRAPH::EdgeIdTag>::EdgeIndicesToContiguousEdgeIndicesImpl;
    };
    







} // namespace detail_graph
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_DETAIL_CONTIGUOUS_INDICES_HXX*/
