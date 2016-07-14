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
        typedef typename Graph:: template EdgeMap<uint8_t>  ContiguousIdStorage;
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



} // namespace detail_graph
} // namespace nifty::graph
} // namespace nifty

#endif /*NIFTY_GRAPH_DETAIL_CONTIGUOUS_INDICES_HXX*/
