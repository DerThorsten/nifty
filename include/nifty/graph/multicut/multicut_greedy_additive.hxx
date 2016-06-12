#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX


#include <map>
#include <functional>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/multicut/multicut_base.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{


    template<class OBJECTIVE>
    class MulticutGreedyAdditive : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
    private:
        typedef detail_graph::UndirectedAdjacency<int64_t,int64_t,int64_t,int64_t> NodeAdjacency;
        typedef std::set<NodeAdjacency> NodeStorage;
        typedef typename Graph:: template NodeMap<NodeStorage> NodesContainer;
        typedef std::pair<int64_t,int64_t> EdgeStorage;
        typedef typename Graph:: template EdgeMap<EdgeStorage> EdgeContainer;
        typedef typename Graph:: template EdgeMap<double> EdgeWeights;
    public:

        struct Settings{

            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
            int verbose { 1 };
        };


        MulticutGreedyAdditive(const Objective & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;

    private:

        void reset();
        bool stopContraction();
        void relabelEdge(const uint64_t edge,const uint64_t deadNode, const uint64_t aliveNode);

        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;

        vigra::ChangeablePriorityQueue< double ,std::greater<double> > pq_;
        NodesContainer nodes_;
        EdgeContainer edges_;
        nifty::ufd::Ufd< > ufd_;
        EdgeWeights weights_;
        uint64_t currentNodeNum_;
    };

    
    template<class OBJECTIVE>
    MulticutGreedyAdditive<OBJECTIVE>::
    MulticutGreedyAdditive(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        pq_(objective.graph().maxEdgeId()+1),
        nodes_(graph_),
        edges_(graph_),
        ufd_(graph_.maxNodeId()+1),
        weights_(graph_),
        currentNodeNum_(graph_.numberOfNodes())
    {
        // do the setup
        this->reset();
    }

    template<class OBJECTIVE>
    void MulticutGreedyAdditive<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        int nodeNum = graph_.numberOfNodes();
        while(!pq_.empty() ){
            
            // get and pop top edge and check 
            // if we are done
            const auto edgeToContract = pq_.top();
            if(settings_.verbose >= 1)
                std::cout<<"node num "<<nodeNum<<" highestWeight "<<pq_.topPriority()<<"\n"; 
            if(this->stopContraction())
                break;
            pq_.pop();

            
            // get the u and v we need to merge into a single node
            const auto uv = edges_[edgeToContract];
            const auto u = uv.first;
            const auto v = uv.second;
            NIFTY_TEST_OP(u,!=,v);

            // merge them into a single node
            ufd_.merge(u, v);
            --nodeNum;

            // check which of u and v is the new representative node
            // also known as 'aliveNode' and which is the deadNode
            const auto aliveNode = ufd_.find(u);
            NIFTY_TEST(aliveNode==u || aliveNode==v);
            const auto deadNode = aliveNode == u ? v : u;      

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
                NIFTY_TEST(pq_.contains(adjToDeadNodeEdge));

                // check if adjToDeadNode is also in 
                // aliveNodes adjacency  => double edge
                const auto findResIter = adjAlive.find(NodeAdjacency(adjToDeadNode));
                if(findResIter != adjAlive.end()){ // we found a double edge

                    NIFTY_TEST_OP(findResIter->node(),==,adjToDeadNode)
                    const auto edgeInAlive = findResIter->edge();
                    NIFTY_TEST(pq_.contains(edgeInAlive));
                    const auto wEdgeInAlive = pq_.priority(edgeInAlive);
                    const auto wEdgeInDead = pq_.priority(adjToDeadNodeEdge);
               
                    // erase the deadNodeEdge 
                    pq_.deleteItem(adjToDeadNodeEdge);
                    pq_.changePriority(edgeInAlive, wEdgeInAlive + wEdgeInDead);

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
        }
        for(auto node : graph_.nodes()){
            nodeLabels[node] = ufd_.find(node);
        }
    }

    template<class OBJECTIVE>
    const typename MulticutGreedyAdditive<OBJECTIVE>::Objective &
    MulticutGreedyAdditive<OBJECTIVE>::
    objective()const{
        return objective_;
    }

    template<class OBJECTIVE>
    inline void 
    MulticutGreedyAdditive<OBJECTIVE>::
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
            NIFTY_TEST(false);
        }
    }

    template<class OBJECTIVE>
    void MulticutGreedyAdditive<OBJECTIVE>::
    reset(
    ){
        ufd_.reset();
        currentNodeNum_ = graph_.numberOfNodes();
        // shortcuts
        

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
        while(!pq_.empty())
            pq_.pop();
        // edges:
        const auto & weights = objective_.weights();
        for(const auto edge: graph_.edges()){
            const auto uv = graph_.uv(edge);
            const auto edgeStorage = EdgeStorage(uv.first, uv.second);
            edges_[edge] = edgeStorage;
            pq_.push(edge, weights[edge]);
        }            
    }

    template<class OBJECTIVE>
    bool  MulticutGreedyAdditive<OBJECTIVE>::
    stopContraction(){
        const auto highestWeight = pq_.topPriority();
        const auto nnsc = settings_.nodeNumStopCond;
        // exit if weight stop cond kicks in
        if(highestWeight < settings_.weightStopCond){
            return true;
        }
        if(nnsc > 0.0){
            uint64_t ns;
            if(nnsc >= 1.0){
                ns = static_cast<uint64_t>(nnsc);
               // if node
            }
            else{
                ns = static_cast<uint64_t>(double(graph_.numberOfNodes())*nnsc +0.5);
            }
        }
        return false;
    }

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX
