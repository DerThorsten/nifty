#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_GREEDY_ADDITIVE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_GREEDY_ADDITIVE_HXX


#include <random>
#include <functional>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"


namespace nifty{
namespace graph{
namespace lifted_multicut{


    namespace detail_lifted_multicut_greedy_additive{

    template<class OBJECTIVE>
    class LiftedMulticutGreedyAdditiveCallback{
    public:

        struct Settings{

            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
            //int verbose { 0 };

            int seed {42};
            bool addNoise {false};
            double sigma{1.0};
        };


        typedef OBJECTIVE Objective;
        typedef typename Objective::LiftedGraph LiftedGraph;
        typedef typename LiftedGraph:: template EdgeMap<double> CurrentWeightMap;
        typedef typename LiftedGraph:: template EdgeMap<bool>   IsLiftedMap;
        typedef vigra::ChangeablePriorityQueue< double ,std::greater<double> > QueueType;

        LiftedMulticutGreedyAdditiveCallback(
            const Objective & objective,
            const Settings & settings
        )
        :   objective_(objective),
            liftedGraph_(objective.liftedGraph()),
            pq_(objective.liftedGraph().edgeIdUpperBound()+1),
            isLifted_(objective.liftedGraph()),
            currentWeight_(objective.liftedGraph()),
            settings_(settings),
            currentNodeNum_(objective.liftedGraph().numberOfNodes()),
            gen_(settings.seed),
            dist_(0.0, settings.sigma){

            this->reset();
        }

        void reset(){

            // reset queue in case something is left
            while(!pq_.empty())
                pq_.pop();

            const auto & weights = objective_.weights();






            objective_.forEachGraphEdge([&](const uint64_t edge){
                isLifted_[edge] = false;
                if(!settings_.addNoise){
                    pq_.push(edge, weights[edge]);
                    currentWeight_[edge] = weights[edge];
                }
                else{
                    const auto w = weights[edge] + dist_(gen_);
                    pq_.push(edge, w);
                    currentWeight_[edge] = w;
                }
            });

            objective_.forEachLiftedeEdge([&](const uint64_t edge){

                // lifted edge
                isLifted_[edge] = true;
                pq_.push(edge, -1.0*std::numeric_limits<double>::infinity());

                if(!settings_.addNoise)
                    currentWeight_[edge] = weights[edge];
                else{
                    currentWeight_[edge] = weights[edge] + dist_(gen_);
                }

            });

        }

        void contractEdge(const uint64_t edgeToContract){
            NIFTY_ASSERT(pq_.contains(edgeToContract));
            NIFTY_CHECK(!isLifted_[edgeToContract],"bug!");
            pq_.deleteItem(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
            --currentNodeNum_;
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){


            NIFTY_ASSERT(pq_.contains(aliveEdge));
            NIFTY_ASSERT(pq_.contains(deadEdge));

            const auto wEdgeInAlive = currentWeight_[aliveEdge];
            const auto wEdgeInDead =  currentWeight_[deadEdge];

            const auto aIsLifted = isLifted_[aliveEdge];
            const auto dIsLifted = isLifted_[deadEdge];

            const auto wSum = wEdgeInAlive + wEdgeInDead;

            // non is lifted, merge as always
            if(!aIsLifted && ! dIsLifted){
                pq_.changePriority(aliveEdge, wSum);
                currentWeight_[aliveEdge] = wSum;
            }
            // both are lifted => merge but keep pq weight at -inf
            else if(aIsLifted &&  dIsLifted){
                currentWeight_[aliveEdge] = wSum;
            }
            // if only the dead edge is lifted
            // we need can merge as always
            else if(!aIsLifted && dIsLifted){
                pq_.changePriority(aliveEdge, wSum);
                currentWeight_[aliveEdge] = wSum;
            }
            // alive edge was lifted, but merged with non lifted
            // which makes the lifted edge a normal edge
            else if(aIsLifted && !dIsLifted){
                pq_.changePriority(aliveEdge, wSum);
                currentWeight_[aliveEdge] = wSum;
                isLifted_[aliveEdge] = false;
            }
            else{
                NIFTY_CHECK(false,"bug");
            }

            pq_.deleteItem(deadEdge);
            

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
                    ns = static_cast<uint64_t>(double(liftedGraph_.numberOfNodes())*nnsc +0.5);
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
        const LiftedGraph & liftedGraph_;
        QueueType pq_;

        IsLiftedMap   isLifted_;
        CurrentWeightMap currentWeight_;

        Settings settings_;
        uint64_t currentNodeNum_;

        std::mt19937 gen_;
        std::normal_distribution<> dist_;
    };

    } // end namespace detail_lifted_multicut_greedy_additive




    template<class OBJECTIVE>
    class LiftedMulticutGreedyAdditive : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef typename Objective::LiftedGraph LiftedGraph;
        typedef detail_lifted_multicut_greedy_additive::LiftedMulticutGreedyAdditiveCallback<Objective> Callback;
        typedef LiftedMulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

    public:

        typedef typename Callback::Settings Settings;

        virtual ~LiftedMulticutGreedyAdditive(){}
        LiftedMulticutGreedyAdditive(const Objective & objective, const Settings & settings = Settings());
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
            return std::string("LiftedMulticutGreedyAdditive");
        }


    private:


        const Objective & objective_;
        const Graph & graph_;
        NodeLabels * currentBest_;

        Callback callback_;
        EdgeContractionGraph<LiftedGraph, Callback> edgeContractionGraph_;
    };

    
    template<class OBJECTIVE>
    LiftedMulticutGreedyAdditive<OBJECTIVE>::
    LiftedMulticutGreedyAdditive(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        currentBest_(nullptr),
        callback_(objective, settings),
        edgeContractionGraph_(objective.liftedGraph(), callback_)
    {
        // do the setup
        this->reset();
    }

    template<class OBJECTIVE>
    void LiftedMulticutGreedyAdditive<OBJECTIVE>::
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
    const typename LiftedMulticutGreedyAdditive<OBJECTIVE>::Objective &
    LiftedMulticutGreedyAdditive<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 
    template<class OBJECTIVE>
    void LiftedMulticutGreedyAdditive<OBJECTIVE>::
    reset(
    ){
        callback_.reset();
        edgeContractionGraph_.reset();
    }

    template<class OBJECTIVE>
    inline void 
    LiftedMulticutGreedyAdditive<OBJECTIVE>::
    changeSettings(
        const Settings & settings
    ){
        callback_.changeSettings(settings);
    }

    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_GREEDY_ADDITIVE_HXX
