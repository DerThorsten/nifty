#pragma once


namespace nifty{
namespace graph{
namespace graph_maps{



/**
 * @brief Implicit edge map
 * @details Convert a node map into an edge map by applying
 * a binary functor to the node maps values.
 * 
 * @tparam GRAPH the graph type
 * @tparam NODE_MAP The node map. This can be a 
 * (const) reference or a value type (in) case of proxy objects
 * @tparam BINARY_FUNCTOR a binary functor.This can be a 
 * (const) reference or a value type.
 */
template<class GRAPH, class NODE_MAP, class BINARY_FUNCTOR>
class EdgeMapFromNodeMap {
public:
    typedef GRAPH GraphType;
    typedef BINARY_FUNCTOR BinaryFunctorType;
    typedef typename BinaryFunctorType::value_type value_type;
    typedef NODE_MAP NodeMapType;

    /**
     * @brief construct edge map from node map and functor  
     * 
     * @param graph the graph       
     * @param nodeMap the node map
     * @param binaryFunctor the binary functor
     */
    EdgeMapFromNodeMap(
        const GraphType & graph,
        NodeMapType nodeMap,
        BinaryFunctorType binaryFunctor
    )
    :   graph_(graph),
        nodeMap_(nodeMap),
        binaryFunctor_(binaryFunctor){
    }

    /**
     * @brief get the value for an edge
     * @details get the value for an edge
     * by calling the binary functor. The functor
     * is called with the node maps values at
     * the enpoints of the edge.
     * 
     * @param edgeIndex the edge index  
     * @return the value of the edge map
     */
    value_type operator[](const uint64_t edgeIndex)const{
        const auto uv = graph_.uv(edgeIndex);
        const auto u = uv.first;
        const auto v = uv.second;
        return binaryFunctor_(nodeMap_[u], nodeMap_[v]);
    }

private:
    const GraphType & graph_;
    NODE_MAP nodeMap_;
    BinaryFunctorType binaryFunctor_;
};







} // namespace nifty::graph::graph_maps
} // namespace nifty::graph
} // namespace nifty

