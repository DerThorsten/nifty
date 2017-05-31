#pragma once

#include <random>
#include <functional>

#include "nifty/tools/changable_priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"


#include "QPBO.h"



namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{

    // \cond SUPPRESS_DOXYGEN

    namespace detail_mincut_greedy_additive{

    template<class OBJECTIVE>
    class MincutGreedyAdditiveCallback{
    public:

        struct Settings{

            double weightStopCond{0.0};
            double nodeNumStopCond{-1};
        

            int seed {42};
            bool addNoise {false};
            double sigma{1.0};
            bool improve{true};
        };


        typedef OBJECTIVE Objective;
        typedef typename Objective::Graph Graph;
        typedef nifty::tools::ChangeablePriorityQueue< double ,std::greater<double> > QueueType;

        MincutGreedyAdditiveCallback(
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
                //std::cout<<"done by weight stop cond\n";
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
                    //std::cout<<"done by node num stop cond\n";
                    return true;
                }
            }
            if(currentNodeNum_<=1){
                //std::cout<<"done by total node stop cond\n";
                return true;
            }
            if(pq_.empty()){
                //std::cout<<"done by empty node stop cond\n";
                return true;
            }
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
        const Settings & settings()const{
            return settings_;
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

    } // end namespace detail_mincut_greedy_additive
    // \endcond 



    template<class OBJECTIVE>
    class MincutGreedyAdditive : public MincutBase<OBJECTIVE>
    {
    public: 
        typedef float QpboValueType;
        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef typename Objective::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef detail_mincut_greedy_additive::MincutGreedyAdditiveCallback<Objective> CallbackType;
        typedef nifty::graph::EdgeContractionGraph<GraphType, CallbackType> ContractionGraphType;
        typedef MincutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

    public:

        typedef typename CallbackType::Settings Settings;

        virtual ~MincutGreedyAdditive(){}
        MincutGreedyAdditive(const Objective & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;

        void reset();
        void changeSettings(const Settings & settings);

        virtual void weightsChanged(){
            this->reset();
        }
        virtual const NodeLabels & currentBestNodeLabels( ){
            //for(auto node : graph_.nodes()){
            //    currentBest_->operator[](node) = edgeContractionGraph_.findRepresentativeNode(node);
            //}
            return *currentBest_;
        }
        virtual double currentBestEnergy() {
           return currentBestEnergy_;
        }
        virtual std::string name()const{
            return std::string("MincutGreedyAdditive");
        }


    private:


        const Objective & objective_;
        const Graph & graph_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;
        CallbackType callback_;
        ContractionGraphType edgeContractionGraph_;
        QPBO<QpboValueType> qpbo_;
    };

    
    template<class OBJECTIVE>
    MincutGreedyAdditive<OBJECTIVE>::
    MincutGreedyAdditive(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        currentBest_(nullptr),
        callback_(objective, settings),
        edgeContractionGraph_(objective.graph(), callback_),
        qpbo_(0,0),
        currentBestEnergy_(std::numeric_limits<double>::infinity())
    {
        // do the setup
        this->reset();
    }

    template<class OBJECTIVE>
    void MincutGreedyAdditive<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){

        VisitorProxy visitorProxy(visitor);
        visitorProxy.addLogNames({"#nodes","topWeight"});
        visitorProxy.begin(this);


        if(graph_.numberOfEdges()>0){

            currentBest_ = & nodeLabels;
            //std::cout<<"a\n";
            // do clustering 
            while(!callback_.done() ){
                //std::cout<<"braa\n";
                // get the edge
                auto edgeToContract = callback_.edgeToContract();

                // contract it
                edgeContractionGraph_.contractEdge(edgeToContract);

            
                
                visitorProxy.setLogValue(0, edgeContractionGraph_.numberOfNodes());
                visitorProxy.setLogValue(1, callback_.queue().topPriority());
                if(!visitorProxy.visit(this)){
                    break;
                }
                
            }

            // do qpbo on rest
            //std::cout<<"b\n";
            const auto & queue = callback_.queue();
            const auto & nodeUfd = edgeContractionGraph_.nodeUfd();
            const auto & nSubNodes = edgeContractionGraph_.numberOfNodes();
            const auto & nSubEdges = edgeContractionGraph_.numberOfEdges();


            //std::cout<<"nSubNodes "<<nSubNodes<<" nSubEdges "<<nSubEdges<<"\n";

            qpbo_.Reset();
            qpbo_.AddNode(nSubNodes);
            qpbo_.SetMaxEdgeNum(nSubEdges);

            // map non dense local ids to dense ids
            uint64_t denseNode = 0;
            typedef typename  GraphType::template  NodeMap<uint64_t> ToDense;
            ToDense toDense(graph_);

            for(const auto node : graph_.nodes()){
                if(nodeUfd.find(node) == node){
                    toDense[node] = denseNode;
                    ++denseNode;
                }
            }

            if(nSubNodes>0){
                qpbo_.AddUnaryTerm(0, 0.0, 100000.0);
            }

            // iterate over all left over edges
            for(const auto edge : graph_.edges()){
                if(queue.contains(edge)){
                    const auto w = queue.priority(edge);
                    const auto u = edgeContractionGraph_.u(edge);
                    const auto v = edgeContractionGraph_.v(edge);
                    const auto subU =toDense[u];
                    const auto subV =toDense[v];
                    //std::cout<<"E "<<u   <<"-"<<v   <<"   "<<w<<"\n";
                    //std::cout<<"  "<<subU<<"-"<<subV<<"   "<<w<<"\n";
                    qpbo_.AddPairwiseTerm(subU, subV, 0.0, w,w, 0.0);
                }
            }

            // Solve 
            //std::cout<<"Solve\n";
            qpbo_.Solve();



            if(this->callback_.settings().improve){
                //std::cout<<"Improve\n";
                qpbo_.Improve();
            }

            auto e = qpbo_.ComputeTwiceEnergy()/2.0;
            if(nSubNodes>0 && qpbo_.GetLabel(0) == 1){
                e -= 100000.0;
            }
            currentBestEnergy_ = e;
            //std::cout<<"qpbo val "<<currentBestEnergy_<<"\n";
            // Map back
            for(const auto node : graph_.nodes()){
                const auto subNode = toDense[edgeContractionGraph_.findRepresentativeNode(node)];

                const auto triLabel = qpbo_.GetLabel(subNode);
                const auto nodeLabel = (triLabel == 0 ? 0 : (triLabel == 1 ? 1 : 0));
                nodeLabels[node] = nodeLabel;
            }
        }
        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename MincutGreedyAdditive<OBJECTIVE>::Objective &
    MincutGreedyAdditive<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 
    template<class OBJECTIVE>
    void MincutGreedyAdditive<OBJECTIVE>::
    reset(
    ){
        callback_.reset();
        edgeContractionGraph_.reset();
    }

    template<class OBJECTIVE>
    inline void 
    MincutGreedyAdditive<OBJECTIVE>::
    changeSettings(
        const Settings & settings
    ){
        callback_.changeSettings(settings);
    }

    
} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
