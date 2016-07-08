#pragma once
#ifndef NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX
#define NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX

#include <iostream>
#include <set>

#include "vigra/multi_array.hxx"
#include "vigra/random_forest.hxx"
#include "vigra/priority_queue.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"
#include <tuple> 

namespace nifty{
namespace graph{



    inline uint64_t myHash( uint64_t u ){
        uint64_t v = u * 3935559000370003845 + 2691343689449507681;

        v ^= v >> 21;
        v ^= v << 37;
        v ^= v >>  4;

        v *= 4768777513237032717;

        v ^= v << 20;
        v ^= v >> 41;
        v ^= v <<  5;

        return v;
    }


    template<class GRAPH, class T, class CLASSIFIER>
    class Gala;


    namespace detail_gala{

    // also the training callback
    template<class GRAPH, class T, class CLASSIFIER>
    struct TrainingCallback{
        
        typedef GRAPH GraphType;
        typedef TrainingCallback<GraphType, T, CLASSIFIER> Self;
        typedef TrainingInstance<GraphType, T>     TrainingInstanceType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;
        typedef  std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> HashType;
        typedef Gala<GraphType, T, CLASSIFIER> GalaType;

        //typedef EdgeContractionGraph<GraphType, Self>   EdgeContractionGraphType;

        typedef EdgeContractionGraphWithSets<GraphType, Self, std::set<uint64_t> >   EdgeContractionGraphType;

        typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;


        typedef typename GraphType:: template EdgeMap<double>  EdgeMapDouble;

        typedef typename GraphType:: template EdgeMap<uint64_t>  EdgeHash;
        typedef typename GraphType:: template NodeMap<uint64_t>  NodeHash;

        TrainingCallback(TrainingInstanceType & trainingInstance, GalaType & gala, const size_t ownIndex)
        :   trainingInstance_(trainingInstance),
            contractionGraph_(trainingInstance.graph(), *this),
            pq_(trainingInstance.graph().maxEdgeId()+1),
            gala_(gala),
            edgeGt_(trainingInstance.graph()),
            edgeGtUncertainty_(trainingInstance.graph()),
            edgeSize_(trainingInstance.graph()),
            edgeHash_(trainingInstance.graph()),
            nodeHash_(trainingInstance.graph()),
            ownIndex_(ownIndex){

            // 
            for(const auto edge : this->graph().edges()){
                edgeGt_[edge] = trainingInstance_.edgeGt()[edge];
                edgeGtUncertainty_[edge] = trainingInstance_.edgeGtUncertainty()[edge];
                edgeSize_[edge] = 1;
                edgeHash_[edge] = myHash(edge);
            }
            for(const auto node : this->graph().nodes()){
                nodeHash_[node] = myHash(node);
            }
        }

        const GraphType & graph()const{
            return trainingInstance_.graph();
        }

        FeatureBaseType * features(){
            return trainingInstance_.features();
        }
        const uint64_t numberOfFeatures()const{
            return trainingInstance_.numberOfFeatures();
        }

        void reset(){
            this->features()->reset();
            contractionGraph_.reset();
            while(!pq_.empty()){
                pq_.pop();
            }
            for(const auto edge : this->graph().edges()){
                edgeGt_[edge] = trainingInstance_.edgeGt()[edge];
                edgeGtUncertainty_[edge] = trainingInstance_.edgeGtUncertainty()[edge];
                edgeSize_[edge] = 1;
                edgeHash_[edge] = myHash(edge);
            };
            for(const auto node : this->graph().nodes()){
                nodeHash_[node] = myHash(node);
            }
        }

        void contractEdge(const uint64_t edgeToContract){
            NIFTY_TEST(pq_.contains(edgeToContract));
            pq_.deleteItem(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
           trainingInstance_.features()->mergeNodes(aliveNode, deadNode);
           nodeHash_[aliveNode] = myHash(nodeHash_[aliveNode] + nodeHash_[deadNode]);
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){


            edgeHash_[aliveEdge] = myHash(edgeHash_[aliveEdge] + edgeHash_[deadEdge]);

            const auto sa = edgeSize_[aliveEdge];
            const auto sd = edgeSize_[deadEdge];
            const auto s = sa + sd;

            edgeSize_[aliveEdge] = s;
            edgeGt_[aliveEdge] = (sa*edgeGt_[aliveEdge] + sd*edgeGt_[deadEdge])/s;
            edgeGtUncertainty_[aliveEdge] = (sa*edgeGtUncertainty_[aliveEdge] + sd*edgeGtUncertainty_[deadEdge])/s;

            trainingInstance_.features()->mergeEdges(aliveEdge, deadEdge);
            pq_.deleteItem(deadEdge);
           
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // recompute features  
            const auto u = contractionGraph_.nodeOfDeadEdge(edgeToContract);
            for(auto adj :contractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                this->recomputeFeaturesAndPredictImpl(edge, true);
            }
        }
        void initalPrediction(){ 
            for(const auto edge: this->graph().edges()){
                this->recomputeFeaturesAndPredictImpl(edge, false);
            }
        }

        void recomputeFeaturesAndPredictImpl(const uint64_t edgeToUpdate, bool useNewExamples){ 

            const auto nf = this->numberOfFeatures();
            std::vector<T> f(nf);
            this->features()->getFeatures(edgeToUpdate, f.data());
            auto pRf = gala_.classifier_.predictProbability(f.data());

            if(useNewExamples){
                const auto labelGt = edgeGt_[edgeToUpdate];
                const auto intLabelGt = labelGt  > 0.5 ? 1 : 0 ;
                const auto labelRf = pRf  > 0.5 ? 1 : 0 ;
                const auto uv = contractionGraph_.uv(edgeToUpdate);
                HashType hash(ownIndex_, edgeHash_[edgeToUpdate],nodeHash_[uv.first],nodeHash_[uv.second]);
                gala_.discoveredExample(f.data(), pRf, labelGt, edgeGtUncertainty_[edgeToUpdate],hash );
            }

            pq_.push(edgeToUpdate, pRf);
        }

        void toContract(std::vector<uint64_t> & toContract){
            toContract.resize(0);
            toContract.push_back(pq_.top());

        }

        TrainingInstanceType & trainingInstance_;
        EdgeContractionGraphType contractionGraph_;
        QueueType pq_;

        EdgeMapDouble edgeGt_;
        EdgeMapDouble edgeGtUncertainty_;
        EdgeMapDouble edgeSize_;

        EdgeHash edgeHash_;
        NodeHash nodeHash_;
        size_t ownIndex_;
        GalaType & gala_;
    };



    template<class GRAPH, class T, class CLASSIFIER>
    struct TestCallback{
        
        typedef GRAPH GraphType;
        typedef TestCallback<GraphType, T, CLASSIFIER> Self;
        typedef Instance<GraphType, T>     InstanceType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;
        typedef Gala<GraphType, T, CLASSIFIER> GalaType;


        typedef EdgeContractionGraphWithSets<GraphType, Self, std::set<uint64_t> >   EdgeContractionGraphType;
        typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

        typedef typename GraphType:: template EdgeMap<double>  EdgeMapDouble;
        typedef typename GraphType:: template EdgeMap<uint64_t>  EdgeHash;
        typedef typename GraphType:: template NodeMap<uint64_t>  NodeHash;

        TestCallback(InstanceType & instance, const GalaType & gala)
        :   instance_(instance),
            contractionGraph_(instance.graph(), *this),
            pq_(instance.graph().maxEdgeId()+1),
            gala_(gala){
        }

        const GraphType & graph()const{
            return instance_.graph();
        }

        FeatureBaseType * features(){
            return instance_.features();
        }
        const uint64_t numberOfFeatures()const{
            return instance_.numberOfFeatures();
        }

        void reset(){
            this->features()->reset();
            contractionGraph_.reset();
            while(!pq_.empty()){
                pq_.pop();
            }
        }

        void contractEdge(const uint64_t edgeToContract){
            NIFTY_TEST(pq_.contains(edgeToContract));
            pq_.deleteItem(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
           instance_.features()->mergeNodes(aliveNode, deadNode);
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
            instance_.features()->mergeEdges(aliveEdge, deadEdge);
            pq_.deleteItem(deadEdge);
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // recompute features  
            const auto u = contractionGraph_.nodeOfDeadEdge(edgeToContract);
            for(auto adj :contractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                this->recomputeFeaturesAndPredictImpl(edge);
            }
        }
        void initalPrediction(){ 
            for(const auto edge: this->graph().edges()){
                this->recomputeFeaturesAndPredictImpl(edge);
            }
        }

        void recomputeFeaturesAndPredictImpl(const uint64_t edgeToUpdate){ 

            const auto nf = this->numberOfFeatures();
            std::vector<T> f(nf);
            this->features()->getFeatures(edgeToUpdate, f.data());
            auto pRf = gala_.classifier_.predictProbability(f.data());
            pq_.push(edgeToUpdate, pRf);
        }

        void toContract(std::vector<uint64_t> & toContract){
            toContract.resize(0);
            toContract.push_back(pq_.top());

        }

        InstanceType & instance_;
        EdgeContractionGraphType contractionGraph_;
        QueueType pq_;
        const GalaType & gala_;
    };


    } // namespace nifty::graph::detail_gala




} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX
