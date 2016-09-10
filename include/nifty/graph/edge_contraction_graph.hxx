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

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD = false>
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
                NIFTY_ASSERT(edgesSet_.find(edgeToContract)!=edgesSet_.end());
                edgesSet_.erase(edgeToContract);
                outerCallback_.contractEdge(edgeToContract);
            }

            void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
                NIFTY_ASSERT(nodesSet_.find(aliveNode)!=nodesSet_.end());
                NIFTY_ASSERT(nodesSet_.find(deadNode)!=nodesSet_.end());
                nodesSet_.erase(deadNode);
                outerCallback_.mergeNodes(aliveNode, deadNode);
            }

            void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
                NIFTY_ASSERT(edgesSet_.find(aliveEdge)!=nodesSet_.end());
                NIFTY_ASSERT(edgesSet_.find(deadEdge)!=nodesSet_.end());
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
        typedef typename CGraphType::NodeUfdType NodeUfdType;

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
        typedef typename TypeHelper::NodeUfdType NodeUfdType;
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
            NIFTY_ASSERT_OP(innerCallback_.nodesSet_.size(),==,cgraph_.numberOfNodes());
            return innerCallback_.nodesSet_.begin();
        }
        NodeIter nodesEnd()const{
            NIFTY_ASSERT_OP(innerCallback_.nodesSet_.size(),==,cgraph_.numberOfNodes());
            return innerCallback_.nodesSet_.end();
        }
        EdgeIter edgesBegin()const{
            NIFTY_ASSERT_OP(innerCallback_.edgesSet_.size(),==,cgraph_.numberOfEdges());
            return innerCallback_.edgesSet_.begin();
        }
        EdgeIter edgesEnd()const{
            NIFTY_ASSERT_OP(innerCallback_.edgesSet_.size(),==,cgraph_.numberOfEdges());
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
            NIFTY_ASSERT(innerCallback_.nodesSet_.find(node)!=innerCallback_.nodesSet_.end());
            return cgraph_.adjacencyBegin(node);
        }
        AdjacencyIter adjacencyEnd(const int64_t node)const{
            NIFTY_ASSERT(innerCallback_.nodesSet_.find(node)!=innerCallback_.nodesSet_.end());
            return cgraph_.adjacencyEnd(node);
        }
        AdjacencyIter adjacencyOutBegin(const int64_t node)const{
            NIFTY_ASSERT(innerCallback_.nodesSet_.find(node)!=innerCallback_.nodesSet_.end());
            return cgraph_.adjacencyOutBegin(node);
        }


        EdgeStorage uv(const uint64_t edge)const{
            NIFTY_ASSERT(innerCallback_.edgesSet_.find(edge)!=innerCallback_.edgesSet_.end());
            return cgraph_.uv(edge);
        }
        int64_t u(const uint64_t edge)const{
            NIFTY_ASSERT(innerCallback_.edgesSet_.find(edge)!=innerCallback_.edgesSet_.end());
            return cgraph_.u(edge);
        }
        int64_t v(const uint64_t edge)const{
            NIFTY_ASSERT(innerCallback_.edgesSet_.find(edge)!=innerCallback_.edgesSet_.end());
            return cgraph_.v(edge);
        }

        uint64_t numberOfNodes()const{
            NIFTY_ASSERT_OP(innerCallback_.nodesSet_.size(),==,cgraph_.numberOfNodes());
            return cgraph_.numberOfNodes();
        }
        uint64_t numberOfEdges()const{
            NIFTY_ASSERT_OP(innerCallback_.edgesSet_.size(),==,cgraph_.numberOfEdges());
            return cgraph_.numberOfEdges();
        }

        int64_t nodeIdUpperBound() const{
            NIFTY_ASSERT_OP(innerCallback_.nodesSet_.size(),==,cgraph_.numberOfNodes());
            return cgraph_.nodeIdUpperBound();
        }
        int64_t edgeIdUpperBound() const{
            NIFTY_ASSERT_OP(innerCallback_.edgesSet_.size(),==,cgraph_.numberOfEdges());
            return cgraph_.edgeIdUpperBound();
        }

        int64_t findEdge(const int64_t u, const int64_t v)const{
            NIFTY_ASSERT(innerCallback_.nodesSet_.find(u)!=innerCallback_.nodesSet_.end());
            NIFTY_ASSERT(innerCallback_.nodesSet_.find(v)!=innerCallback_.nodesSet_.end());
            return cgraph_.findEdge(u, v);
        }

        
        void contractEdge(const uint64_t edgeToContract){
            NIFTY_ASSERT(innerCallback_.edgesSet_.find(edgeToContract)!=innerCallback_.edgesSet_.end());
            cgraph_.contractEdge(edgeToContract);
        }
        void reset(){
            cgraph_.reset();
            innerCallback_.initSets();
        }
        NodeUfdType & ufd(){
            return cgraph_.ufd();
        } 
        const NodeUfdType & ufd() const{
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
            NIFTY_ASSERT(innerCallback_.edgesSet_.find(deadEdge)==innerCallback_.edgesSet_.end());
            return cgraph_.nodeOfDeadEdge(deadEdge);
        }
    private:
        typedef detail_edge_contraction_graph::InnerCallback<GraphType, OuterCallbackType, SetType> InnerCallbackType;
        InnerCallbackType innerCallback_;
        CGraphType cgraph_;
    };









    namespace detail_edge_contraction_graph{


        template<class GRAPH, bool WITH_EDGE_UFD>
        class EdgeContractionGraphEdgeUfdHelper;


        template<class GRAPH>
        class EdgeContractionGraphEdgeUfdHelper<GRAPH, true>{
        public:
            EdgeContractionGraphEdgeUfdHelper(const GRAPH & graph){
            }
        protected:
            std::pair<uint64_t, uint64_t> edgeUfdMerge(uint64_t alive, uint64_t dead){
                return std::pair<uint64_t, uint64_t>(alive, dead);
            };
        private:

        };

        template<class GRAPH>
        class EdgeContractionGraphEdgeUfdHelper<GRAPH, false>{
        private:
            typedef nifty::ufd::Ufd< > EdgeUfdType;
        public:
            EdgeContractionGraphEdgeUfdHelper(const GRAPH & graph)
            :   edgeUfd_(graph.edgeIdUpperBound()+1){
            }

        protected:
            std::pair<uint64_t, uint64_t> edgeUfdMerge(uint64_t edge1, uint64_t edge2){
                edgeUfd_.merge(edge1, edge2);
                const auto alive = edgeUfd_.find(edge1);
                const auto dead = (alive == edge1 ? edge2 : edge1);
                return std::pair<uint64_t, uint64_t>(alive, dead);
            };

        private:
            EdgeUfdType edgeUfd_;
        };

    }




    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    class EdgeContractionGraph : 
        public 
        detail_edge_contraction_graph::EdgeContractionGraphEdgeUfdHelper<GRAPH, WITH_EDGE_UFD>

    {
    public:
        typedef GRAPH Graph;
        typedef CALLBACK Callback;
        typedef nifty::ufd::Ufd< > NodeUfdType;
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
        NodeUfdType & ufd(); // is this a good idea to have this public
                         // 
        const NodeUfdType & ufd() const;
        const Graph & baseGraph()const;
        const Graph & graph()const;
        uint64_t findRepresentativeNode(const uint64_t node)const;
        uint64_t findRepresentativeNode(const uint64_t node);
        uint64_t nodeOfDeadEdge(const uint64_t deadEdge)const;
            

    private:

        void relabelEdge(const uint64_t edge,const uint64_t deadNode, const uint64_t aliveNode);

        const Graph & graph_;

        Callback & callback_;

        NodesContainer nodes_;
        EdgeContainer edges_;
        NodeUfdType nodeUfd_;
        uint64_t currentNodeNum_;
        uint64_t currentEdgeNum_;
    };


    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    EdgeContractionGraph(
        const Graph & graph,   
        Callback & callback
    )
    :   detail_edge_contraction_graph::EdgeContractionGraphEdgeUfdHelper<GRAPH, WITH_EDGE_UFD>(graph),
        graph_(graph),
        callback_(callback),
        nodes_(graph_),
        edges_(graph_),
        nodeUfd_(graph_.nodeIdUpperBound()+1),
        currentNodeNum_(graph_.numberOfNodes()),
        currentEdgeNum_(graph_.numberOfEdges())
    {
        this->reset();
    }


    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::AdjacencyIterRange 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    adjacency(
        const int64_t node
    ) const{
        return AdjacencyIterRange(adjacencyBegin(node),adjacencyEnd(node));
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    adjacencyBegin(
        const int64_t node
    )const{
        return nodes_[node].begin();
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    adjacencyEnd(
        const int64_t node
    )const{
        return nodes_[node].end();
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::AdjacencyIter 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    adjacencyOutBegin(
        const int64_t node
    )const{
        return adjacencyBegin(node);
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>    
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::EdgeStorage 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    uv(
        const uint64_t edge
    )const{
        return edges_[edge];
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>    
    inline int64_t
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    u(
        const uint64_t edge
    )const{
        return edges_[edge].first;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>    
    inline int64_t
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    v(
        const uint64_t edge
    )const{
        return edges_[edge].second;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline const typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::NodeUfdType & 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    ufd() const {
        return nodeUfd_;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::NodeUfdType & 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    ufd() {
        return nodeUfd_;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    numberOfNodes()const{
        return currentNodeNum_;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    numberOfEdges()const{
        return currentEdgeNum_;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    nodeIdUpperBound()const{
        return graph_.nodeIdUpperBound();
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    edgeIdUpperBound()const{
        return graph_.edgeIdUpperBound();
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline int64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
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


    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    reset(){
        nodeUfd_.reset();
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

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
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
        NIFTY_ASSERT_OP(nodeUfd_.find(u),==,u);
        NIFTY_ASSERT_OP(nodeUfd_.find(v),==,v);
        nodeUfd_.merge(u, v);
        --currentNodeNum_;

        // check which of u and v is the new representative node
        // also known as 'aliveNode' and which is the deadNode
        const auto aliveNode = nodeUfd_.find(u);
        NIFTY_ASSERT(aliveNode==u || aliveNode==v);
        const auto deadNode = aliveNode == u ? v : u;       
        NIFTY_ASSERT_OP(nodeUfd_.find(aliveNode),==,aliveNode);
        NIFTY_ASSERT_OP(nodeUfd_.find(deadNode),!=,deadNode);

        callback_.mergeNodes(aliveNode, deadNode);


        // get the adjacency sets of both nodes
        auto & adjAlive = nodes_[aliveNode];
        auto & adjDead = nodes_[deadNode];
        
        // remove them from each other
        // 
        // 
        NIFTY_ASSERT(adjAlive.find(NodeAdjacency(deadNode)) != adjAlive.end());
        NIFTY_ASSERT(adjDead.find(NodeAdjacency(aliveNode)) != adjDead.end());

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
                

                if(!WITH_EDGE_UFD){

                        callback_.mergeEdges(edgeInAlive, adjToDeadNodeEdge);
                       

                        
                }
                else{
                    const auto ret = this->edgeUfdMerge(edgeInAlive, adjToDeadNodeEdge);
                    const auto aliveEdge = ret.first;
                    const auto deadEdge = ret.second;
                    if(aliveEdge == edgeInAlive){
                        callback_.mergeEdges(edgeInAlive, adjToDeadNodeEdge);
                    }
                    else{
                        auto & uv = edges_[aliveEdge];
                        uv.first = aliveNode;
                        uv.second = deadNode;

                        nodes_[aliveNode].find(NodeAdjacency(adjToDeadNode))->changeEdgeIndex(aliveEdge);
                        nodes_[adjToDeadNode].find(NodeAdjacency(aliveNode))->changeEdgeIndex(aliveEdge);
                    }
                }   
                // relabel adjacency
                --currentEdgeNum_;
                auto & s = nodes_[adjToDeadNode];
                auto findRes = s.find(NodeAdjacency(deadNode));
                s.erase(NodeAdjacency(deadNode));

            }
            else{   // no double edge
                // shift adjacency from dead to alive
                adjAlive.insert(NodeAdjacency(adjToDeadNode, adjToDeadNodeEdge));

                // relabel adjacency 
                auto & s = nodes_[adjToDeadNode];

                NIFTY_ASSERT(s.find(NodeAdjacency(deadNode)) != s.end());
                NIFTY_ASSERT(s.find(NodeAdjacency(aliveNode)) == s.end());
                
                s.erase(NodeAdjacency(deadNode));
                s.insert(NodeAdjacency(aliveNode, adjToDeadNodeEdge));
                // relabel edge
                this->relabelEdge(adjToDeadNodeEdge, deadNode, aliveNode);
            }
        }

        callback_.contractEdgeDone(edgeToContract);
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    findRepresentativeNode(
        const uint64_t node
    )const{
        return nodeUfd_.find(node);
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    findRepresentativeNode(
        const uint64_t node
    ){
        return nodeUfd_.find(node);
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline uint64_t 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    nodeOfDeadEdge(
        const uint64_t deadEdge
    )const{
        auto uv = edges_[deadEdge];
        NIFTY_TEST_OP(nodeUfd_.find(uv.first),==, nodeUfd_.find(uv.second));
        return nodeUfd_.find(uv.first);
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline const typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::Graph & 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    baseGraph()const{
        return graph_;
    }

    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline const typename EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::Graph & 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
    graph()const{
        return graph_;
    }


    template<class GRAPH, class CALLBACK, bool WITH_EDGE_UFD>
    inline void 
    EdgeContractionGraph<GRAPH, CALLBACK, WITH_EDGE_UFD>::
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
