#pragma once


#include "nifty/graph/graph_tags.hxx"

namespace nifty{
namespace graph{
namespace detail_graph{

    template<class GRAPH, class EDGE_ID_TAG>
    class EdgeIndicesToContiguousEdgeIndicesImpl{
    public:
        typedef GRAPH GraphType;
        EdgeIndicesToContiguousEdgeIndicesImpl(const GraphType & graph)
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
        typedef typename GraphType:: template EdgeMap<uint64_t>  ContiguousIdStorage;
        ContiguousIdStorage ids_;
    };


    template<class GRAPH>
    class EdgeIndicesToContiguousEdgeIndicesImpl<GRAPH, nifty::graph::ContiguousTag>  {
    public:
        typedef GRAPH GraphType;
        EdgeIndicesToContiguousEdgeIndicesImpl(const GraphType & graph){
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
        typedef GRAPH GraphType;
        NodeIndicesToContiguousNodeIndicesImpl(const GraphType & graph)
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
        typedef typename GraphType:: template NodeMap<uint64_t>  ContiguousIdStorage;
        ContiguousIdStorage ids_;
    };


    template<class GRAPH>
    class NodeIndicesToContiguousNodeIndicesImpl<GRAPH, nifty::graph::ContiguousTag>  {
    public:
        typedef GRAPH GraphType;
        NodeIndicesToContiguousNodeIndicesImpl(const GraphType & graph){
        }
        int64_t operator[](const int64_t node)const{
            return node;
        }
    private:
    };


    

    template<class GRAPH>
    class NodeIndicesToContiguousNodeIndices 
    :   public  NodeIndicesToContiguousNodeIndicesImpl<GRAPH, typename GRAPH::NodeIdTag>{
    public:
        using NodeIndicesToContiguousNodeIndicesImpl<GRAPH, typename GRAPH::NodeIdTag>::NodeIndicesToContiguousNodeIndicesImpl;
    };
    







} // namespace detail_graph
} // namespace nifty::graph
} // namespace nifty

