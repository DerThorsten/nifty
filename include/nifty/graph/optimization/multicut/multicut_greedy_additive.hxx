#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX


#include <random>
#include <functional>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"

//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{



    namespace detail_multicut_greedy_additive{

    template<class OBJECTIVE>
    class MulticutGreedyAdditiveCallback{
    public:

        struct Settings{

            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
    

            int seed {42};
            bool addNoise {false};
            double sigma{1.0};
        };


        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef vigra::ChangeablePriorityQueue< double ,std::greater<double> > QueueType;

        MulticutGreedyAdditiveCallback(
            const Objective & objective,
            const Settings & settings
        )
        :   objective_(objective),
            graph_(objective.graph()),
            pq_(objective.graph().edgeIdUpperBound()+1 ),
            settings_(settings),
            currentNodeNum_(objective.graph().numberOfNodes()),
            gen_(settings.seed),
            dist_(0.0, settings.sigma){

            this->reset();
        }

        void reset(){

            // reset queue in case something is left
            while(!pq_.empty())
                pq_.pop();

            const auto & weights = objective_.weights();
            for(const auto edge: graph_.edges()){
                if(!settings_.addNoise)
                    pq_.push(edge, weights[edge]);
                else{
                    pq_.push(edge, weights[edge] + dist_(gen_));
                }
            }    
        }

        void contractEdge(const uint64_t edgeToContract){
            NIFTY_ASSERT(pq_.contains(edgeToContract));
            pq_.deleteItem(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
            --currentNodeNum_;
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
            NIFTY_ASSERT(pq_.contains(aliveEdge));
            NIFTY_ASSERT(pq_.contains(deadEdge));
            const auto wEdgeInAlive = pq_.priority(aliveEdge);
            const auto wEdgeInDead = pq_.priority(deadEdge);
            pq_.deleteItem(deadEdge);
            pq_.changePriority(aliveEdge, wEdgeInAlive + wEdgeInDead);
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // nothing to do here
        }
        bool done(){
            const auto highestWeight = pq_.topPriority();
            const auto nnsc = settings_.nodeNumStopCond;
            // exit if weight stop cond kicks in
            if(highestWeight <= settings_.weightStopCond){
                return true;
            }
            if(nnsc > 0.0){
                uint64_t ns;
                if(nnsc >= 1.0){
                    ns = static_cast<uint64_t>(nnsc);
                }
                else{
                    ns = static_cast<uint64_t>(double(graph_.numberOfNodes())*nnsc +0.5);
                }
                if(currentNodeNum_ <= ns){
                    return true;
                }
            }
            if(currentNodeNum_<=1)
                return true;
            if(pq_.empty())
                return true;
            return false;
        }

        uint64_t edgeToContract(){
            return pq_.top();
        }

        void changeSettings(
            const Settings & settings
        ){
            settings_ = settings;
        }

        const QueueType & queue()const{
            return pq_;
        }

    private:

        const Objective & objective_;
        const Graph & graph_;
        QueueType pq_;
        Settings settings_;
        uint64_t currentNodeNum_;

        std::mt19937 gen_;
        std::normal_distribution<> dist_;
    };

    } // end namespace detail_multicut_greedy_additive




    template<class OBJECTIVE>
    class MulticutGreedyAdditive : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef detail_multicut_greedy_additive::MulticutGreedyAdditiveCallback<Objective> Callback;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

    public:

        typedef typename Callback::Settings Settings;

        virtual ~MulticutGreedyAdditive(){}
        MulticutGreedyAdditive(const Objective & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;

        void reset();
        void changeSettings(const Settings & settings);

        virtual void weightsChanged(){
            this->reset();
        }
        virtual const NodeLabels & currentBestNodeLabels( ){
            for(auto node : graph_.nodes()){
                currentBest_->operator[](node) = edgeContractionGraph_.findRepresentativeNode(node);
            }
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("MulticutGreedyAdditive");
        }


    private:


        const Objective & objective_;
        const Graph & graph_;
        NodeLabels * currentBest_;

        Callback callback_;
        EdgeContractionGraph<Graph, Callback> edgeContractionGraph_;
    };

    
    template<class OBJECTIVE>
    MulticutGreedyAdditive<OBJECTIVE>::
    MulticutGreedyAdditive(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        currentBest_(nullptr),
        callback_(objective, settings),
        edgeContractionGraph_(objective.graph(), callback_)
    {
        // do the setup
        this->reset();
    }

    template<class OBJECTIVE>
    void MulticutGreedyAdditive<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        

        if(visitor!=nullptr){
            visitor->addLogNames({"#nodes","topWeight"});
            visitor->begin(this);
        }
        if(graph_.numberOfEdges()>0){
            currentBest_ = & nodeLabels;
            while(!callback_.done() ){
                    
                // get the edge
                auto edgeToContract = callback_.edgeToContract();

                // contract it
                edgeContractionGraph_.contractEdge(edgeToContract);

            
                if(visitor!=nullptr){
                   visitor->setLogValue(0, edgeContractionGraph_.numberOfNodes());
                   visitor->setLogValue(1, callback_.queue().topPriority());
                   if(!visitor->visit(this)){
                       break;
                   }
                }
            }
            
            for(auto node : graph_.nodes()){
                nodeLabels[node] = edgeContractionGraph_.findRepresentativeNode(node);
            }
        }
        if(visitor!=nullptr)
            visitor->end(this);
    }

    template<class OBJECTIVE>
    const typename MulticutGreedyAdditive<OBJECTIVE>::Objective &
    MulticutGreedyAdditive<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 
    template<class OBJECTIVE>
    void MulticutGreedyAdditive<OBJECTIVE>::
    reset(
    ){
        callback_.reset();
        edgeContractionGraph_.reset();
    }

    template<class OBJECTIVE>
    inline void 
    MulticutGreedyAdditive<OBJECTIVE>::
    changeSettings(
        const Settings & settings
    ){
        callback_.changeSettings(settings);
    }

    

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_GREEDY_ADDITIVE_HXX
