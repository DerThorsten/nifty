#pragma once
#ifndef NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX
#define NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif

#include "nifty/graph/simple_graph.hxx"
#include "nifty/container/flat_set.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"

//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{

    template<class GRAPH, class CALLBACK>
    class EdgeContractionGraph;

    template<class GRAPH, class OUTER_CALLBACK, class SET>
    class EdgeContractionGraphWithSets;

    namespace detail_edge_contraction_graph{

        template<class GRAPH, class OUTER_CALLBACK, class SET>
        class InnerCallback{
        public:
            typedef GRAPH GraphType;
            typedef OUTER_CALLBACK OuterCallbackType;
            typedef SET SetType;
            InnerCallback(const GraphType & g, OuterCallbackType & outerCallback)
            :   graph_(g),
                outerCallback_(outerCallback){
                this->initSets();
            }
            void initSets(){
                edgesSet_.clear();
                nodesSet_.clear();
                for(const auto edge : graph_.edges()){
                    edgesSet_.insert(edge);
                }
                for(const auto node : graph_.nodes()){
                    nodesSet_.insert(node);
                }
            }
            void reset(){
                this->initSets();
                outerCallback_.reset();
            }

            void contractEdge(const uint64_t edgeToContract){
                edgesSet_.erase(edgeToContract);
                outerCallback_.contractEdge(edgeToContract);
            }

            void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
                nodesSet_.erase(deadNode);
                outerCallback_.mergeNodes(aliveNode, deadNode);
            }

            void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
                edgesSet_.erase(deadEdge);
                outerCallback_.mergeEdges(aliveEdge, deadEdge);
            }

            void contractEdgeDone(const uint64_t edgeToContract){
                outerCallback_.contractEdgeDone(edgeToContract);
            }


        private:
            const GraphType & graph_;
            OuterCallbackType & outerCallback_;
            SetType nodesSet_;
            SetType edgesSet_;


        };

    }



    template<class GRAPH, class OUTER_CALLBACK, class SET>
    class EdgeContractionGraphWithSets :
        public EdgeContractionGraph<GRAPH, detail_edge_contraction_graph::InnerCallback<GRAPH, OUTER_CALLBACK, SET> >{
    public:
        typedef EdgeContractionGraphWithSets<GRAPH, OUTER_CALLBACK, SET> SelfType;
        typedef EdgeContractionGraph<GRAPH, detail_edge_contraction_graph::InnerCallback<GRAPH, OUTER_CALLBACK, SET> > BaseType;
        typedef GRAPH GraphType;
        typedef OUTER_CALLBACK OuterCallbackType;
        typedef SET SetType;
        
        EdgeContractionGraphWithSets(const GraphType & graph, OuterCallbackType & outerCallback)
        :   innerCallback_(graph, outerCallback),
            BaseType(graph, innerCallback_){
                innerCallback_.initSets();
        }

    private:
        typedef detail_edge_contraction_graph::InnerCallback<GraphType, OuterCallbackType, SetType> InnerCallbackType;
        InnerCallbackType innerCallback_;
    };



    template<class GRAPH, class CALLBACK>
    class EdgeContractionGraph{
        public:
            typedef GRAPH Graph;
            typedef CALLBACK Callback;
            typedef nifty::ufd::Ufd< > UfdType;
        private:
            typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,int64_t,int64_t> NodeAdjacency;
            //typedef std::set<NodeAdjacency> NodeStorage;
            typedef nifty::container::FlatSet <NodeAdjacency> NodeStorage;
            typedef typename NodeStorage::const_iterator AdjacencyIter;

            typedef typename Graph:: template NodeMap<NodeStorage> NodesContainer;
            typedef std::pair<int64_t,int64_t> EdgeStorage;
            typedef typename Graph:: template EdgeMap<EdgeStorage> EdgeContainer;
        public:

            EdgeContractionGraph(const Graph & graph,   Callback & callback)
            :
                graph_(graph),
                callback_(callback),
                nodes_(graph_),
                edges_(graph_),
                ufd_(graph_.maxNodeId()+1),
                currentNodeNum_(graph_.numberOfNodes()),
                currentEdgeNum_(graph_.numberOfEdges())
            {
                this->reset();
            }

            struct AdjacencyIterRange :  public tools::ConstIteratorRange<AdjacencyIter>{
                using tools::ConstIteratorRange<AdjacencyIter>::ConstIteratorRange;
            };
            AdjacencyIterRange adjacency(const int64_t node) const{
                return AdjacencyIterRange(adjacencyBegin(node),adjacencyEnd(node));
            }

            AdjacencyIter adjacencyBegin(const int64_t node)const{
                NIFTY_ASSERT_OP(node,<,numberOfNodes());
                return nodes_[node].begin();
            }

            AdjacencyIter adjacencyEnd(const int64_t node)const{
                NIFTY_ASSERT_OP(node,<,numberOfNodes());
                return nodes_[node].end();
            }

            AdjacencyIter adjacencyOutBegin(const int64_t node)const{
                return adjacencyBegin(node);
            }


            void reset(){
                ufd_.reset();
                currentNodeNum_ = graph_.numberOfNodes();
                currentEdgeNum_ = graph_.numberOfEdges();
                

                // fill the data-structures for the dynamic graph
                //  nodes:
                for(const auto u : graph_.nodes()){
                    auto & dAdj = nodes_[u];
                    dAdj.clear();
                    for(const auto adj : graph_.adjacency(u)){
                        const auto v = adj.node();
                        const auto edge = adj.edge();
                        dAdj.insert(NodeAdjacency(v, edge));
                    }
                }

                
                // edges:
                for(const auto edge: graph_.edges()){
                    const auto uv = graph_.uv(edge);
                    const auto edgeStorage = EdgeStorage(uv.first, uv.second);
                    edges_[edge] = edgeStorage;
                }            
            }

            void contractEdge(const uint64_t edgeToContract){


                // 
                callback_.contractEdge(edgeToContract);
                --currentEdgeNum_;
                
                // get the u and v we need to merge into a single node
                const auto uv = edges_[edgeToContract];
                const auto u = uv.first;
                const auto v = uv.second;
                NIFTY_ASSERT_OP(u,!=,v);

                // merge them into a single node
                ufd_.merge(u, v);
                --currentNodeNum_;

                // check which of u and v is the new representative node
                // also known as 'aliveNode' and which is the deadNode
                const auto aliveNode = ufd_.find(u);
                NIFTY_ASSERT(aliveNode==u || aliveNode==v);
                const auto deadNode = aliveNode == u ? v : u;      



                callback_.mergeNodes(aliveNode, deadNode);


                // get the adjacency sets of both nodes
                auto & adjAlive = nodes_[aliveNode];
                auto & adjDead = nodes_[deadNode];
                
                // remove them from each other
                adjAlive.erase(NodeAdjacency(deadNode));
                adjDead.erase(NodeAdjacency(aliveNode));


                // we will "shift/move" the adj. nodes
                // from 'adjDead' into 'adjAlive':
                for(auto adj : adjDead){

                    const auto adjToDeadNode = adj.node();
                    const auto adjToDeadNodeEdge = adj.edge();


                    // check if adjToDeadNode is also in 
                    // aliveNodes adjacency  => double edge
                    const auto findResIter = adjAlive.find(NodeAdjacency(adjToDeadNode));
                    if(findResIter != adjAlive.end()){ // we found a double edge

                        NIFTY_ASSERT_OP(findResIter->node(),==,adjToDeadNode)
                        const auto edgeInAlive = findResIter->edge();
                            //NIFTY_ASSERT(pq_.contains(edgeInAlive));
                                //  const auto wEdgeInAlive = pq_.priority(edgeInAlive);
                                //  const auto wEdgeInDead = pq_.priority(adjToDeadNodeEdge);
                   
                        // erase the deadNodeEdge 
                                //  pq_.deleteItem(adjToDeadNodeEdge);
                                //  pq_.changePriority(edgeInAlive, wEdgeInAlive + wEdgeInDead);
                        
                        callback_.mergeEdges(edgeInAlive, adjToDeadNodeEdge);
                        --currentEdgeNum_;

                        // relabel adjacency
                        auto & s = nodes_[adjToDeadNode];
                        auto findRes = s.find(NodeAdjacency(deadNode));
                        s.erase(NodeAdjacency(deadNode));
                    }
                    else{   // no double edge
                        // shift adjacency from dead to alive
                        adjAlive.insert(NodeAdjacency(adjToDeadNode, adjToDeadNodeEdge));

                        // relabel adjacency 
                        auto & s = nodes_[adjToDeadNode];
                        s.erase(NodeAdjacency(deadNode));
                        s.insert(NodeAdjacency(aliveNode, adjToDeadNodeEdge));
                        // relabel edge
                        this->relabelEdge(adjToDeadNodeEdge, deadNode, aliveNode);
                    }
                }

                callback_.contractEdgeDone(edgeToContract);
            }

            EdgeStorage uv(const uint64_t edge)const{
                return edges_[edge];
            }

            const UfdType & ufd() const {
                return ufd_;
            }

            UfdType & ufd() {
                return ufd_;
            }

            uint64_t numberOfNodes()const{
                return currentNodeNum_;
            }
            uint64_t numberOfEdges()const{
                return currentEdgeNum_;
            }

            uint64_t findRepresentativeNode(const uint64_t node)const{
                return ufd_.find(node);
            }
            uint64_t findRepresentativeNode(const uint64_t node){
                return ufd_.find(node);
            }

            uint64_t nodeOfDeadEdge(const uint64_t deadEdge)const{
                auto uv = edges_[deadEdge];
                NIFTY_TEST_OP(ufd_.find(uv.first),==, ufd_.find(uv.second));
                return ufd_.find(uv.first);
            }
            const Graph & baseGraph()const{
                return graph_;
            }
        private:

            void relabelEdge(const uint64_t edge,const uint64_t deadNode, const uint64_t aliveNode){
                auto & uv = edges_[edge];
                if(uv.first == deadNode){
                    uv.first = aliveNode;
                }
                else if(uv.second == deadNode){
                    uv.second = aliveNode;
                }
                else{
                    NIFTY_ASSERT(false);
                } 
            }


            const Graph & graph_;

            Callback & callback_;

            NodesContainer nodes_;
            EdgeContainer edges_;
            UfdType ufd_;
            uint64_t currentNodeNum_;
            uint64_t currentEdgeNum_;
    };


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX
