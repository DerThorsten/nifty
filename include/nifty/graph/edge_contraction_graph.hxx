#pragma once
#ifndef NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX
#define NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX

#include <functional>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif

#include "nifty/graph/undirected_graph_base.hxx"
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




    struct FlexibleCallback{
        inline void contractEdge(const uint64_t edgeToContract){
            if(contractEdgeCallback)
                contractEdgeCallback(edgeToContract);
        }

        inline void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
            if(mergeNodesCallback)
                mergeNodesCallback(aliveNode, deadNode);
        }

        inline void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
            if(mergeEdgesCallback)
                mergeEdgesCallback(aliveEdge, deadEdge);
        }

        inline void contractEdgeDone(const uint64_t edgeToContract){
            if(contractEdgeDoneCallback)
                contractEdgeDoneCallback(edgeToContract);
        }
        std::function<void(uint64_t) >          contractEdgeCallback;
        std::function<void(uint64_t,uint64_t) > mergeNodesCallback;
        std::function<void(uint64_t,uint64_t) > mergeEdgesCallback;
        std::function<void(uint64_t) >          contractEdgeDoneCallback;
    };  








    namespace detail_edge_contraction_graph{

        template<class GRAPH, class OUTER_CALLBACK, class SET>
        struct InnerCallback{
        //public:
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


        //private:
            const GraphType & graph_;
            OuterCallbackType & outerCallback_;
            SetType nodesSet_;
            SetType edgesSet_;


        };

        struct ForwardIterableThing{

            typedef uint64_t KeyType;


            ForwardIterableThing(const uint64_t maxId = 0)
            :   idToNextId_(maxId+1,-1),
                beginId_(-1),
                endId_(-1),
                currentSize_(0){

            }


            bool contains(const KeyType & key) const{
                return idToNextId_[key];
            }

            uint64_t size()const{
                currentSize_;
            }

            
            
            std::vector<int64_t> idToNextId_;
            int64_t beginId_;
            int64_t endId_; 
            uint64_t currentSize_;           
        };

    }

    template<class GRAPH, class OUTER_CALLBACK, class SET>
    struct EdgeContractionGraphWithSetsHelper{
        typedef GRAPH GraphType;
        typedef OUTER_CALLBACK OuterCallbackType;
        typedef SET SetType;
        typedef EdgeContractionGraphWithSets<GraphType, OuterCallbackType, SET> SelfType;
        typedef detail_edge_contraction_graph::InnerCallback<GraphType, OuterCallbackType, SetType> InnerCallbackType;
        typedef EdgeContractionGraph<GRAPH, InnerCallbackType > CGraphType;
        typedef typename CGraphType::AdjacencyIter AdjacencyIter;
        typedef typename CGraphType::EdgeStorage EdgeStorage;
        typedef typename CGraphType::UfdType UfdType;

        typedef typename SetType::const_iterator EdgeIter;
        typedef typename SetType::const_iterator NodeIter;
    }; 



    template<class GRAPH, class OUTER_CALLBACK, class SET>
    class EdgeContractionGraphWithSets : public
    UndirectedGraphBase<
        EdgeContractionGraphWithSets<GRAPH, OUTER_CALLBACK, SET>,
        typename EdgeContractionGraphWithSetsHelper<GRAPH, OUTER_CALLBACK, SET>::NodeIter,
        typename EdgeContractionGraphWithSetsHelper<GRAPH, OUTER_CALLBACK, SET>::EdgeIter,
        typename EdgeContractionGraphWithSetsHelper<GRAPH, OUTER_CALLBACK, SET>::AdjacencyIter
    >
    {

        typedef EdgeContractionGraphWithSetsHelper<GRAPH,OUTER_CALLBACK,SET> TypeHelper;

        typedef typename TypeHelper::SelfType SelfType;
        typedef typename TypeHelper::CGraphType CGraphType;
        typedef typename TypeHelper::EdgeStorage EdgeStorage;
    public:
        typedef typename TypeHelper::GraphType GraphType;
        typedef typename TypeHelper::OuterCallbackType OuterCallbackType;
        typedef typename TypeHelper::SetType SetType;
        typedef typename TypeHelper::UfdType UfdType;
        typedef typename TypeHelper::EdgeIter EdgeIter;
        typedef typename TypeHelper::NodeIter NodeIter;
        typedef typename TypeHelper::AdjacencyIter AdjacencyIter;

        typedef SparseTag EdgeIdTag;
        typedef SparseTag NodeIdTag;

        typedef SortedTag EdgeIdOrderTag;
        typedef SortedTag NodeIdOrderTag;


        
        EdgeContractionGraphWithSets(const GraphType & graph, OuterCallbackType & outerCallback)
        :   innerCallback_(graph, outerCallback),
            cgraph_(graph, innerCallback_){
                innerCallback_.initSets();
        }


        NodeIter nodesBegin()const{
            return innerCallback_.nodesSet_.begin();
        }
        NodeIter nodesEnd()const{
            return innerCallback_.nodesSet_.end();
        }
        EdgeIter edgesBegin()const{
            return innerCallback_.edgesSet_.begin();
        }
        EdgeIter edgesEnd()const{
            return innerCallback_.edgesSet_.end();
        }


        template<class F>
        void forEachEdge(F && f)const{
            for(const auto edge : innerCallback_.edgesSet_){
                f(edge);
            }
        }
        template<class F>
        void forEachNode(F && f)const{
            for(const auto node : innerCallback_.nodesSet_){
                f(node);
            }
        }
 
        AdjacencyIter adjacencyBegin(const int64_t node)const{
            return cgraph_.adjacencyBegin(node);
        }
        AdjacencyIter adjacencyEnd(const int64_t node)const{
            return cgraph_.adjacencyEnd(node);
        }
        AdjacencyIter adjacencyOutBegin(const int64_t node)const{
            return cgraph_.adjacencyOutBegin(node);
        }


        EdgeStorage uv(const uint64_t edge)const{
            return cgraph_.uv(edge);
        }
        int64_t u(const uint64_t edge)const{
            return cgraph_.u(edge);
        }
        int64_t v(const uint64_t edge)const{
            return cgraph_.v(edge);
        }

        uint64_t numberOfNodes()const{
            return cgraph_.numberOfNodes();
        }
        uint64_t numberOfEdges()const{
            return cgraph_.numberOfEdges();
        }

        uint64_t nodeIdUpperBound() const{
            return cgraph_.nodeIdUpperBound();
        }
        uint64_t edgeIdUpperBound() const{
            return cgraph_.edgeIdUpperBound();
        }

        int64_t findEdge(const int64_t u, const int64_t v)const{
            return cgraph_.findEdge(u, v);
        }

        
        void contractEdge(const uint64_t edgeToContract){
            cgraph_.contractEdge(edgeToContract);
        }
        void reset(){
            cgraph_.reset();
            innerCallback_.initSets();
        }
        UfdType & ufd(){
            return cgraph_.ufd();
        } 
        const UfdType & ufd() const{
            return cgraph_.ufd();
        }
        const GraphType & baseGraph()const{
            return cgraph_.baseGraph();
        }
        uint64_t findRepresentativeNode(const uint64_t node)const{
            return cgraph_.findRepresentativeNode(node);
        }
        uint64_t findRepresentativeNode(const uint64_t node){
            return cgraph_.findRepresentativeNode(node);
        }
        uint64_t nodeOfDeadEdge(const uint64_t deadEdge)const{
            return cgraph_.nodeOfDeadEdge(deadEdge);
        }


    private:
        typedef detail_edge_contraction_graph::InnerCallback<GraphType, OuterCallbackType, SetType> InnerCallbackType;
        InnerCallbackType innerCallback_;
        CGraphType cgraph_;
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
        
    public:
        typedef std::pair<int64_t,int64_t> EdgeStorage;
        typedef typename NodeStorage::const_iterator AdjacencyIter;
    private:
        typedef typename Graph:: template NodeMap<NodeStorage> NodesContainer; 
        typedef typename Graph:: template EdgeMap<EdgeStorage> EdgeContainer;
    public:

        EdgeContractionGraph(const Graph & graph,   Callback & callback);

        struct AdjacencyIterRange :  public tools::ConstIteratorRange<AdjacencyIter>{
            using tools::ConstIteratorRange<AdjacencyIter>::ConstIteratorRange;
        };

        AdjacencyIterRange adjacency(const int64_t node) const;
        AdjacencyIter adjacencyBegin(const int64_t node)const;
        AdjacencyIter adjacencyEnd(const int64_t node)const;
        AdjacencyIter adjacencyOutBegin(const int64_t node)const;


        EdgeStorage uv(const uint64_t edge)const;
        int64_t u(const uint64_t edge)const;
        int64_t v(const uint64_t edge)const;

        uint64_t numberOfNodes()const;
        uint64_t numberOfEdges()const;

        uint64_t nodeIdUpperBound() const;
        uint64_t edgeIdUpperBound() const;
        
        int64_t findEdge(const int64_t u, const int64_t v)const;


        void contractEdge(const uint64_t edgeToContract);
        void reset();
        UfdType & ufd(); // is this a good idea to have this public
                         // 
        const UfdType & ufd() const;
        const Graph & baseGraph()const;
        uint64_t findRepresentativeNode(const uint64_t node)const;
        uint64_t findRepresentativeNode(const uint64_t node);
        uint64_t nodeOfDeadEdge(const uint64_t deadEdge)const;
        

    private:

        void relabelEdge(const uint64_t edge,const uint64_t deadNode, const uint64_t aliveNode);

        const Graph & graph_;

        Callback & callback_;

        NodesContainer nodes_;
        EdgeContainer edges_;
        UfdType ufd_;
        uint64_t currentNodeNum_;
        uint64_t currentEdgeNum_;
    };



    template<class GRAPH, class CALLBACK>
    inline EdgeContractionGraph<GRAPH, CALLBACK>::
    EdgeContractionGraph(
        const Graph & graph,   
        Callback & callback
    )
    :
        graph_(graph),
        callback_(callback),
        nodes_(graph_),
        edges_(graph_),
        ufd_(graph_.nodeIdUpperBound()+1),
        currentNodeNum_(graph_.numberOfNodes()),
        currentEdgeNum_(graph_.numberOfEdges())
    {
        this->reset();
    }


    template<class GRAPH, class CALLBACK>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::AdjacencyIterRange 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    adjacency(
        const int64_t node
    ) const{
        return AdjacencyIterRange(adjacencyBegin(node),adjacencyEnd(node));
    }

    template<class GRAPH, class CALLBACK>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    adjacencyBegin(
        const int64_t node
    )const{
        return nodes_[node].begin();
    }

    template<class GRAPH, class CALLBACK>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    adjacencyEnd(
        const int64_t node
    )const{
        return nodes_[node].end();
    }

    template<class GRAPH, class CALLBACK>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    adjacencyOutBegin(
        const int64_t node
    )const{
        return adjacencyBegin(node);
    }

    template<class GRAPH, class CALLBACK>    
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::EdgeStorage 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    uv(
        const uint64_t edge
    )const{
        return edges_[edge];
    }

    template<class GRAPH, class CALLBACK>    
    inline int64_t
    EdgeContractionGraph<GRAPH, CALLBACK>::
    u(
        const uint64_t edge
    )const{
        return edges_[edge].first;
    }

    template<class GRAPH, class CALLBACK>    
    inline int64_t
    EdgeContractionGraph<GRAPH, CALLBACK>::
    v(
        const uint64_t edge
    )const{
        return edges_[edge].second;
    }

    template<class GRAPH, class CALLBACK>
    inline const typename EdgeContractionGraph<GRAPH, CALLBACK>::UfdType & 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    ufd() const {
        return ufd_;
    }

    template<class GRAPH, class CALLBACK>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK>::UfdType & 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    ufd() {
        return ufd_;
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    numberOfNodes()const{
        return currentNodeNum_;
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    numberOfEdges()const{
        return currentEdgeNum_;
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    nodeIdUpperBound()const{
        return graph_.nodeIdUpperBound();
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    edgeIdUpperBound()const{
        return graph_.edgeIdUpperBound();
    }

    template<class GRAPH, class CALLBACK>
    inline int64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    findEdge(
        const int64_t u, 
        const int64_t v
    )const{
        const auto fres =  nodes_[u].find(NodeAdjacency(v));
        if(fres != nodes_[u].end())
            return fres->edge();
        else
            return -1;
    }


    template<class GRAPH, class CALLBACK>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    reset(){
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

    template<class GRAPH, class CALLBACK>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    contractEdge(
        const uint64_t edgeToContract
    ){

        // 
        callback_.contractEdge(edgeToContract);
        --currentEdgeNum_;
        
        // get the u and v we need to merge into a single node
        const auto uv = edges_[edgeToContract];
        const auto u = uv.first;
        const auto v = uv.second;
        NIFTY_TEST_OP(u,!=,v);

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

                NIFTY_TEST_OP(findResIter->node(),==,adjToDeadNode)
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

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    findRepresentativeNode(
        const uint64_t node
    )const{
        return ufd_.find(node);
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    findRepresentativeNode(
        const uint64_t node
    ){
        return ufd_.find(node);
    }

    template<class GRAPH, class CALLBACK>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    nodeOfDeadEdge(
        const uint64_t deadEdge
    )const{
        auto uv = edges_[deadEdge];
        NIFTY_TEST_OP(ufd_.find(uv.first),==, ufd_.find(uv.second));
        return ufd_.find(uv.first);
    }

    template<class GRAPH, class CALLBACK>
    inline const typename EdgeContractionGraph<GRAPH, CALLBACK>::Graph & 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    baseGraph()const{
        return graph_;
    }

    template<class GRAPH, class CALLBACK>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK>::
    relabelEdge(
        const uint64_t edge,
        const uint64_t deadNode, 
        const uint64_t aliveNode
    ){
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


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_EDGE_CONTRACTION_GRAPH_HXX
