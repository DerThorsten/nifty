#pragma once
#ifndef NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX
#define NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX

#include <iostream>
#include <set>
#include <tuple> 

#include "vigra/multi_array.hxx"
#include "vigra/random_forest.hxx"
#include "vigra/priority_queue.hxx"
#include "vigra/algorithm.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"
#include "nifty/graph/gala/gala_instance.hxx"
#include "nifty/graph/gala/contraction_order.hxx"



#include "nifty/graph/multicut/multicut_base.hxx"
#include "nifty/graph/multicut/multicut_visitor_base.hxx"
#include "nifty/graph/multicut/multicut_factory.hxx"
#include "nifty/graph/multicut/fusion_move_based.hxx"
#include "nifty/graph/multicut/multicut_greedy_additive.hxx"
#include "nifty/graph/multicut/proposal_generators/watershed_proposals.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/multicut/perturb_and_map.hxx"




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


    template<class GRAPH, class CGRAPH>
    class MulticutEdgeOrder{

    public:
        typedef nifty::graph::UndirectedGraph<> McOrderGraph;
        typedef nifty::graph::MulticutObjective<McOrderGraph, double> McOrderObjective;
        typedef nifty::graph::MulticutFactoryBase<McOrderObjective> McOrderFactoryBaseType;
        typedef std::shared_ptr<McOrderFactoryBaseType> McFactory;

        MulticutEdgeOrder(
            const GRAPH & graph, 
            const CGRAPH & cgraph,
            McFactory mapFactory,
            McFactory perturbAndMapFactory
        )
        :   graph_(graph),
            cgraph_(cgraph),
            mapFactory_(mapFactory),
            perturbAndMapFactory_(perturbAndMapFactory){

        }


        template<class PROBS_IN>
        double run(
            const PROBS_IN & probsIn,
            std::vector<uint64_t> & toContract,
            bool train
        ){

            const auto & graph = graph_;
            const auto & cgraph = cgraph_;

            typedef typename GRAPH:: template NodeMap<uint64_t>  MgToDense;
            const auto currentNodeNum = cgraph.numberOfNodes();
            const auto currentEdgeNum = cgraph.numberOfEdges();
            MgToDense sparseToDense(graph);
            std::vector<uint64_t> denseToSparse(currentNodeNum);

            uint64_t denseNode = 0;
            cgraph.forEachNode([&](const uint64_t sparseNode){
                NIFTY_CHECK_OP(cgraph.ufd().find(sparseNode),==, sparseNode,"");
                sparseToDense[sparseNode] = denseNode;
                denseToSparse[denseNode] = sparseNode;
                ++denseNode;
            });

            typedef nifty::graph::UndirectedGraph<> DenseGraph;
            typedef nifty::graph::MulticutObjective<DenseGraph, double> Objective;
            typedef nifty::graph::MulticutFactoryBase<Objective> FactoryBaseType;
            typedef nifty::graph::PerturbAndMap<Objective>   PerturbAndMapType;
            typedef typename PerturbAndMapType::Settings PerturbAndMapSettingsType;
            typedef typename PerturbAndMapType::EdgeState PerturbAndMapEdgeState;
            typedef typename PerturbAndMapType::NodeLabels PerturbAndMapNodeLabels;
            DenseGraph denseGraph(currentNodeNum, currentEdgeNum);
            std::vector<double> pBuffer(currentEdgeNum);

            uint64_t denseEdge = 0;
            
            std::vector<int> denseEdgeToSparse(currentEdgeNum);
            cgraph.forEachEdge([&](const uint64_t sparseEdge){
                const auto uvSparse = cgraph.uv(sparseEdge);
                const auto uDense = sparseToDense[uvSparse.first];
                const auto vDense = sparseToDense[uvSparse.second];
                pBuffer[denseEdge] = probsIn[sparseEdge];
                NIFTY_CHECK_OP(uDense,!=, vDense,"");
                auto de = denseGraph.insertEdge(uDense, vDense);
                NIFTY_CHECK_OP(de,==, denseEdge,"");
                denseEdgeToSparse[denseEdge] = sparseEdge;
                ++denseEdge;
            });

            Objective objective(denseGraph);



            


            for(const auto edge : denseGraph.edges()){
                auto p1 = pBuffer[edge];
                if(p1 > 1.5){
                    objective.weights()[edge] = - 9000000000.0;
                }
                else{
                    p1 = std::min(p1, 0.999999999);
                    p1 = std::max(p1, 0.000000001);

                    const auto p0 = 1.0 - p1;
                    const auto w = std::log(p0/p1);
                    //std::cout<<"p1 "<<p1<<" w "<<w<<"\n";
                    objective.weights()[edge] = w;
                }
            }




            auto solver = mapFactory_->createSharedPtr(objective);
            PerturbAndMapNodeLabels startingPoint(denseGraph);
            solver->optimize(startingPoint, nullptr);


            PerturbAndMapSettingsType s;
            s.mcFactory = perturbAndMapFactory_;
            s.numberOfIterations = 100;
            s.numberOfThreads = -1;
            s.noiseMagnitude = 15.0;
            s.noiseType = PerturbAndMapType::UNIFORM_NOISE;
            PerturbAndMapType pAndMap(objective, s);

            PerturbAndMapEdgeState edgeState(denseGraph);





            //for(const auto edge : denseGraph.edges()){
            //    const auto uv = denseGraph.uv(edge);
            //    const auto es = int(startingPoint[uv.first]!=startingPoint[uv.second]);
            //    //std::cout<<"map state "<<es<<"\n"; 
            //} 
             
            pAndMap.optimize(startingPoint, edgeState);

            
            std::vector<int> sortedIndices(currentEdgeNum);
            for( size_t i=0; i<currentEdgeNum; ++i){
                if(pBuffer[i] > 1.5){
                    edgeState[i] = 2.0;
                }
                sortedIndices[i] = i;
            }





            vigra::indexSort(edgeState.begin(), edgeState.end(), sortedIndices.begin());


            for(size_t i=0; i< std::min(size_t(1), size_t(currentEdgeNum)); ++i){
                const auto de = sortedIndices[i];
                const auto se = denseEdgeToSparse[de];
                toContract.push_back(se);
                std::cout<<se<<" "<<edgeState[de]<<  "\n";
            }
            return edgeState[sortedIndices[0]];
            //throw std::runtime_error("\n");

        }

    private:
        const GRAPH & graph_;
        const CGRAPH & cgraph_;
        McFactory mapFactory_;
        McFactory perturbAndMapFactory_;
    };

   





    // also the training callback
    template<class GRAPH, class T, class CLASSIFIER>
    struct TrainingCallback{
        
        typedef T ValueType;
        typedef GRAPH GraphType;
        typedef TrainingCallback<GraphType, T, CLASSIFIER> Self;
        typedef McGreedyHybridBase<Self> ContractionOrder;
        typedef TrainingInstance<GraphType, T>     TrainingInstanceType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;
        typedef  std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> HashType;
        typedef Gala<GraphType, T, CLASSIFIER> GalaType;



        //typedef EdgeContractionGraph<GraphType, Self>   EdgeContractionGraphType;

        typedef EdgeContractionGraphWithSets<GraphType, Self, std::set<uint64_t> >   EdgeContractionGraphType;

        typedef MulticutEdgeOrder<GraphType, EdgeContractionGraphType> McOrder;



        typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;


        typedef typename GraphType:: template EdgeMap<double>  EdgeMapDouble;

        typedef typename GraphType:: template EdgeMap<uint64_t>  EdgeHash;
        typedef typename GraphType:: template NodeMap<uint64_t>  NodeHash;

        TrainingCallback(TrainingInstanceType & trainingInstance, GalaType & gala, const size_t ownIndex)
        :   trainingInstance_(trainingInstance),
            contractionGraph_(trainingInstance.graph(), *this),
            gala_(gala),
            edgeGt_(trainingInstance.graph()),
            edgeGtUncertainty_(trainingInstance.graph()),
            edgeSizes_(trainingInstance.graph()),
            edgeHash_(trainingInstance.graph()),
            nodeHash_(trainingInstance.graph()),
            ownIndex_(ownIndex),
            mcOrder_(trainingInstance.graph(), contractionGraph_,
                     gala.trainingSettings_.mapFactory,
                     gala.trainingSettings_.perturbAndMapFactory),
            contractionOrder_(*this,true)
            {

            // 
            for(const auto edge : this->graph().edges()){
                edgeGt_[edge] = trainingInstance_.edgeGt()[edge];
                edgeGtUncertainty_[edge] = trainingInstance_.edgeGtUncertainty()[edge];
                edgeSizes_[edge] = getInstance().edgeSizes()[edge];
                edgeHash_[edge] = myHash(edge);
            }
            for(const auto node : this->graph().nodes()){
                nodeHash_[node] = myHash(node);
            }

        }

        const EdgeMapDouble & currentEdgeSizes()const{
            return edgeSizes_;
        }

        TrainingInstanceType & getInstance(){
            return trainingInstance_;
        }

        const EdgeContractionGraphType & cgraph()const{
            return contractionGraph_;
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
            contractionOrder_.reset();
            for(const auto edge : this->graph().edges()){
                edgeGt_[edge] = trainingInstance_.edgeGt()[edge];
                edgeGtUncertainty_[edge] = trainingInstance_.edgeGtUncertainty()[edge];
                edgeSizes_[edge] = getInstance().edgeSizes()[edge];
                edgeHash_[edge] = myHash(edge);
            };
            for(const auto node : this->graph().nodes()){
                nodeHash_[node] = myHash(node);
            }

        }

        void contractEdge(const uint64_t edgeToContract){
            contractionOrder_.contractEdge(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
           trainingInstance_.features()->mergeNodes(aliveNode, deadNode);
           nodeHash_[aliveNode] = myHash(nodeHash_[aliveNode] + nodeHash_[deadNode]);
           contractionOrder_.mergeNodes(aliveNode, deadNode);
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){


            edgeHash_[aliveEdge] = myHash(edgeHash_[aliveEdge] + edgeHash_[deadEdge]);

            const auto sa = edgeSizes_[aliveEdge];
            const auto sd = edgeSizes_[deadEdge];
            const auto s = sa + sd;


            trainingInstance_.features()->mergeEdges(aliveEdge, deadEdge);
            contractionOrder_.mergeEdges(aliveEdge, deadEdge);


            edgeSizes_[aliveEdge] = s;
            edgeGt_[aliveEdge] = (sa*edgeGt_[aliveEdge] + sd*edgeGt_[deadEdge])/s;
            edgeGtUncertainty_[aliveEdge] = (sa*edgeGtUncertainty_[aliveEdge] + sd*edgeGtUncertainty_[deadEdge])/s;

           
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // recompute features  
            const auto u = contractionGraph_.nodeOfDeadEdge(edgeToContract);
            //NIFTY_TEST_OP(u,<,contractionGraph_.numberOfNodes());
            for(auto adj :contractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                const auto p = this->recomputeFeaturesAndPredictImpl(edge, true);
            }
            contractionOrder_.contractEdgeDone(edgeToContract);
        }
        void initalPrediction(){ 
            for(const auto edge: this->graph().edges()){
                const auto p = this->recomputeFeaturesAndPredictImpl(edge, false);
                contractionOrder_.setInitalLocalRfProb(edge, p);
            }
        }

        T recomputeFeaturesAndPredictImpl(const uint64_t edgeToUpdate, bool useNewExamples){ 

            const auto nf = this->numberOfFeatures();
            std::vector<T> f(nf);
            this->features()->getFeatures(edgeToUpdate, f.data());
            const auto p = gala_.classifier_.predictProbability(f.data());
            contractionOrder_.updateLocalRfProb(edgeToUpdate, p);

            if(useNewExamples){
                const auto labelGt = edgeGt_[edgeToUpdate];
                const auto intLabelGt = labelGt  > 0.5 ? 1 : 0 ;
                const auto labelRf = p  > 0.5 ? 1 : 0 ;
                const auto uv = contractionGraph_.uv(edgeToUpdate);
                HashType hash(ownIndex_, edgeHash_[edgeToUpdate],nodeHash_[uv.first],nodeHash_[uv.second]);
                gala_.discoveredExample(f.data(), p, labelGt, edgeGtUncertainty_[edgeToUpdate],hash );
            }
            return p;
        }

        uint64_t edgeToContractNext(){
            return contractionOrder_.edgeToContractNext();
        }
        bool stopContraction(){
            return contractionOrder_.stopContraction();
        }
        TrainingInstanceType & trainingInstance_;
        EdgeContractionGraphType contractionGraph_;

        EdgeMapDouble edgeGt_;
        EdgeMapDouble edgeGtUncertainty_;
        EdgeMapDouble edgeSizes_;

        EdgeHash edgeHash_;
        NodeHash nodeHash_;
        size_t ownIndex_;
        GalaType & gala_;
        McOrder mcOrder_;
        ContractionOrder contractionOrder_;
    };



    template<class GRAPH, class T, class CLASSIFIER>
    struct TestCallback{
        
        typedef T ValueType;
        typedef GRAPH GraphType;
        typedef TestCallback<GraphType, T, CLASSIFIER> Self;
        typedef McGreedyHybridBase<Self> ContractionOrder;
        typedef Instance<GraphType, T>     InstanceType;
        typedef GalaFeatureBase<GraphType, T>     FeatureBaseType;
        typedef Gala<GraphType, T, CLASSIFIER> GalaType;


        typedef EdgeContractionGraphWithSets<GraphType, Self, std::set<uint64_t> >   EdgeContractionGraphType;
        typedef MulticutEdgeOrder<GraphType, EdgeContractionGraphType> McOrder;


        typedef vigra::ChangeablePriorityQueue< double ,std::less<double> > QueueType;

        typedef typename GraphType:: template EdgeMap<double>  EdgeMapDouble;
        typedef typename GraphType:: template EdgeMap<uint64_t>  EdgeHash;
        typedef typename GraphType:: template NodeMap<uint64_t>  NodeHash;

        TestCallback(InstanceType & instance, const GalaType & gala)
        :   instance_(instance),
            contractionGraph_(instance.graph(), *this),
            edgeSizes_(instance.graph()),
            gala_(gala),
            mcOrder_(instance.graph(), contractionGraph_,
                     gala.trainingSettings_.mapFactory,
                     gala.trainingSettings_.perturbAndMapFactory),
            contractionOrder_(*this,false){

            for(const auto edge : this->graph().edges()){
                edgeSizes_[edge] = getInstance().edgeSizes()[edge];
            };

        }

        const EdgeMapDouble & currentEdgeSizes()const{
            return edgeSizes_;
        }

        InstanceType & getInstance(){
            return instance_;
        }

        const EdgeContractionGraphType & cgraph()const{
            return contractionGraph_;
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
            for(const auto edge : this->graph().edges()){
                edgeSizes_[edge] = getInstance().edgeSizes()[edge];
            };
        }

        void contractEdge(const uint64_t edgeToContract){
            contractionOrder_.contractEdge(edgeToContract);
        }

        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
           instance_.features()->mergeNodes(aliveNode, deadNode);
           contractionOrder_.mergeNodes(aliveNode, deadNode);
        }

        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
            instance_.features()->mergeEdges(aliveEdge, deadEdge);
            contractionOrder_.mergeEdges(aliveEdge, deadEdge);
        }

        void contractEdgeDone(const uint64_t edgeToContract){
            // recompute features  
            const auto u = contractionGraph_.nodeOfDeadEdge(edgeToContract);
            for(auto adj :contractionGraph_.adjacency(u)){
                const auto edge = adj.edge();
                const auto p = this->recomputeFeaturesAndPredictImpl(edge);
            }
            contractionOrder_.contractEdgeDone(edgeToContract);
        }
        void initalPrediction(){ 
            for(const auto edge: this->graph().edges()){
                const auto p = this->recomputeFeaturesAndPredictImpl(edge);
                contractionOrder_.setInitalLocalRfProb(edge, p);
            }
        }

        T recomputeFeaturesAndPredictImpl(const uint64_t edgeToUpdate){ 

            const auto nf = this->numberOfFeatures();
            std::vector<T> f(nf);
            this->features()->getFeatures(edgeToUpdate, f.data());
            auto p = gala_.classifier_.predictProbability(f.data());
            return p;
        }

        uint64_t edgeToContractNext(){
            return contractionOrder_.edgeToContractNext();
        }
        bool stopContraction(){
            return contractionOrder_.stopContraction();
        }

        InstanceType & instance_;
        EdgeContractionGraphType contractionGraph_;
        EdgeMapDouble edgeSizes_;

        const GalaType & gala_;
        McOrder mcOrder_;
        ContractionOrder contractionOrder_;
    };


    } // namespace nifty::graph::detail_gala




} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_DETAIL_GALA_CONTRACT_EDGE_CALLBACK_HXX
