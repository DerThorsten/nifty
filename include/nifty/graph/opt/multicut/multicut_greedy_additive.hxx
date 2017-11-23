#pragma once

#include <random>
#include <functional>


#include "nifty/tools/changable_priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"

//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{
namespace opt{
namespace multicut{
    // \cond SUPPRESS_DOXYGEN

    namespace detail_multicut_greedy_additive{

    template<class OBJECTIVE>
    class MulticutGreedyAdditiveCallback{
    public:

        struct SettingsType{

            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
    

            int seed {42};
            bool addNoise {false};
            double sigma{1.0};

            int visitNth{1};
        };


        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef nifty::tools::ChangeablePriorityQueue< double ,std::greater<double> > QueueType;

        MulticutGreedyAdditiveCallback(
            const ObjectiveType & objective,
            const SettingsType & settings
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
            if(currentNodeNum_<=1){
                return true;
            }
            if(pq_.empty()){
                return true;
            }
            return false;
        }

        uint64_t edgeToContract(){
            return pq_.top();
        }

        void changeSettings(
            const SettingsType & settings
        ){
            settings_ = settings;
        }

        const QueueType & queue()const{
            return pq_;
        }   

        const SettingsType & settings()const{
            return settings_;
        }

    private:

        const ObjectiveType & objective_;
        const GraphType & graph_;
        QueueType pq_;
        SettingsType settings_;
        uint64_t currentNodeNum_;

        std::mt19937 gen_;
        std::normal_distribution<> dist_;
    };

    } // end namespace detail_multicut_greedy_additive
    // \endcond 



    template<class OBJECTIVE>
    class MulticutGreedyAdditive : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef detail_multicut_greedy_additive::MulticutGreedyAdditiveCallback<ObjectiveType> Callback;
        typedef MulticutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;

    public:

        typedef typename Callback::SettingsType SettingsType;

        virtual ~MulticutGreedyAdditive(){}
        MulticutGreedyAdditive(const ObjectiveType & objective, const SettingsType & settings = SettingsType());
        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;

        void reset();
        void changeSettings(const SettingsType & settings);

        virtual void weightsChanged(){
            this->reset();
        }
        virtual const NodeLabelsType & currentBestNodeLabels( ){
            for(auto node : graph_.nodes()){
                currentBest_->operator[](node) = edgeContractionGraph_.findRepresentativeNode(node);
            }
            return *currentBest_;
        }
        
        virtual std::string name()const{
            return std::string("MulticutGreedyAdditive");
        }


    private:


        const ObjectiveType & objective_;
        const GraphType & graph_;
        NodeLabelsType * currentBest_;

        Callback callback_;
        EdgeContractionGraph<GraphType, Callback> edgeContractionGraph_;
    };

    
    template<class OBJECTIVE>
    MulticutGreedyAdditive<OBJECTIVE>::
    MulticutGreedyAdditive(
        const ObjectiveType & objective, 
        const SettingsType & settings
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
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){
        

        if(visitor!=nullptr){
            visitor->addLogNames({"#nodes","topWeight"});
            visitor->begin(this);
        }
        auto c = 1;
        if(graph_.numberOfEdges()>0){
            currentBest_ = & nodeLabels;
            while(!callback_.done() ){
                    
                // get the edge
                auto edgeToContract = callback_.edgeToContract();

                // contract it
                edgeContractionGraph_.contractEdge(edgeToContract);

                
                if(c % callback_.settings().visitNth == 0){

                    if(visitor!=nullptr){
                       visitor->setLogValue(0, edgeContractionGraph_.numberOfNodes());
                       visitor->setLogValue(1, callback_.queue().topPriority());
                       if(!visitor->visit(this)){
                            std::cout<<"end by visitor\n";
                           break;
                       }
                    }
                }
                ++c;
            }
            
            for(auto node : graph_.nodes()){
                nodeLabels[node] = edgeContractionGraph_.findRepresentativeNode(node);
            }
        }
        if(visitor!=nullptr)
            visitor->end(this);
    }

    template<class OBJECTIVE>
    const typename MulticutGreedyAdditive<OBJECTIVE>::ObjectiveType &
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
        const SettingsType & settings
    ){
        callback_.changeSettings(settings);
    }

} // namespace nifty::graph::opt::multicut   
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
