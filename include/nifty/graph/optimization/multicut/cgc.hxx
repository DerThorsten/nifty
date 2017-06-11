#pragma once

#include <queue>

#include "boost/format.hpp"

#include "nifty/tools/changable_priority_queue.hxx"


#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/common/solver_factory.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"


#include "nifty/graph/optimization/mincut/mincut_visitor_base.hxx"
#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/graph/optimization/mincut/mincut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"



namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{
    

    /// \cond HIDDEN_SYMBOLS
    namespace detail_cgc{

        template<class OBJECTIVE>
        class SubmodelOptimizer{
        public:
            typedef OBJECTIVE ObjectiveType;
            typedef typename ObjectiveType::WeightType WeightType;
            typedef MulticutBase<ObjectiveType> MulticutBaseType;
            typedef typename ObjectiveType::Graph GraphType;
            typedef typename ObjectiveType::WeightsMap WeightsMapType;
            typedef typename GraphType:: template NodeMap<uint64_t> GlobalNodeToLocal;
            typedef std::vector<uint64_t>                       LocalNodeToGlobal;

            typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;


            typedef nifty::graph::UndirectedGraph<>           SubGraph;
            typedef mincut::MincutObjective<SubGraph, double>         MincutSubObjective;
            typedef mincut::MincutBase<MincutSubObjective>            MincutSubBase;
            typedef nifty::graph::optimization::common::SolverFactoryBase<MincutSubBase>   MincutSubMcFactoryBase;
            
            typedef mincut::MincutVerboseVisitor<MincutSubObjective>  SubMcVerboseVisitor;
     
            typedef typename  MincutSubBase::NodeLabels     MincutSubNodeLabels;

            struct Optimzie1ReturnType{
                Optimzie1ReturnType(const bool imp, const double val)
                :   improvment(imp),
                    minCutValue(val){
                }
                bool        improvment;
                double      minCutValue;
            };

            struct Optimzie2ReturnType{
                Optimzie2ReturnType(const bool imp, const double val)
                :   improvment(imp),
                    improvedBy(val){
                }
                bool        improvment;
                double      improvedBy;
            };

            SubmodelOptimizer(
                const ObjectiveType  & objective, 
                IsDirtyEdge & isDirtyEdge,
                std::shared_ptr<MincutSubMcFactoryBase> & mincutFactory
            )
            :   objective_(objective),
                graph_(objective.graph()),
                weights_(objective.weights()),
                globalNodeToLocal_(objective.graph()),
                localNodeToGlobal_(objective.graph().numberOfNodes()),
                nLocalNodes_(0),
                nLocalEdges_(0),
                ufd_(),
                isDirtyEdge_(isDirtyEdge),
                mincutFactory_(mincutFactory),
                insideEdges_(),
                borderEdges_()
            {
                isDirtyEdge_.reserve(graph_.numberOfNodes());
                borderEdges_.reserve(graph_.numberOfNodes()/4);
                if(!bool(mincutFactory)){
                    throw std::runtime_error("Cgc mincutFactory shall not be empty");
                }
            }

            template<class NODE_LABELS, class ANCHOR_QUEUE>
            Optimzie1ReturnType optimize1(
                NODE_LABELS & nodeLabels,
                const uint64_t anchorNode,
                ANCHOR_QUEUE & anchorQueue
            ){
                // get mapping from local to global and vice versa
                // also counts nLocalEdges
                const auto maxNodeLabel = this->varMapping(nodeLabels, anchorNode);

                //std::cout<<"nLocalNodes "<<nLocalNodes_<<"\n";
                //for(auto localNode=0; localNode<nLocalNodes_; ++localNode){
                //    std::cout<<"uLocal "<<localNode<<" "<<localNodeToGlobal_[localNode]<<"\n";
                //}
            
                ufd_.assign(nLocalNodes_);

                if(nLocalNodes_ >= 2){


                    const auto anchorLabel = nodeLabels[anchorNode];

                    // setup the submodel
                    SubGraph        subGraph(nLocalNodes_);
                    MincutSubObjective    subObjective(subGraph);
                    auto &          subWeights = subObjective.weights();
                    //NIFTY_CHECK_OP(0,==,subWeights.size(),"");


                    this->forEachInternalEdge(nodeLabels, anchorLabel,[&](const uint64_t uLocal, const uint64_t vLocal, const uint64_t edge){

                        const auto w = weights_[edge];
                        //NIFTY_CHECK_OP(subGraph.findEdge(uLocal, vLocal),==,-1,"");
                        const auto edgeId = subGraph.insertEdge(uLocal, vLocal);
                        // NIFTY_CHECK_OP(edgeId,==,subWeights.size(),"");
                        subWeights.insertedEdges(edgeId);
                        subWeights[edgeId] = w;
                    });
                    // solve it
                    MincutSubNodeLabels subgraphRes(subGraph);
                    auto solverPtr = mincutFactory_->create(subObjective);

                    //SubMcVerboseVisitor visitor;
                    solverPtr->optimize(subgraphRes,nullptr);
                    const auto minCutValue  = subObjective.evalNodeLabels(subgraphRes);
                    delete solverPtr;   


                    Optimzie1ReturnType res(false,minCutValue);

                    if(minCutValue < 0.0){
                        //std::cout<<"minCutValue "<<minCutValue<<"\n";

                        this->forEachInternalEdge(nodeLabels, anchorLabel,[&](const uint64_t uLocal, const uint64_t vLocal, const uint64_t edge){
                            if(subgraphRes[uLocal] == subgraphRes[vLocal]){
                                ufd_.merge(uLocal, vLocal);
                            }
                        });
                        NIFTY_CHECK_OP(ufd_.numberOfSets(),>=,2,"");
                        res.improvment = true;
                        res.minCutValue = minCutValue;
                        std::unordered_map<uint64_t,uint64_t> mapping;
                        ufd_.representativeLabeling(mapping);

                        std::vector<uint64_t> anchors(mapping.size());
                        std::vector<uint64_t> anchorsSize(mapping.size(),0);

                        for(auto localNode=0; localNode<nLocalNodes_; ++localNode){
                            const auto node = localNodeToGlobal_[localNode];
                            const auto denseLocalLabel =  mapping[ufd_.find(localNode)];
                            anchorsSize[denseLocalLabel] +=1;
                            anchors[denseLocalLabel] = node;
                            nodeLabels[node] = denseLocalLabel + maxNodeLabel + 1;
                            //std::cout<<" res localNode "<<localNode<<" "<<nodeLabels[node]<<"\n";
                        }
                        //std::cout<<"---\n";
                        for(auto i=0; i<anchors.size(); ++i){
                            if(anchorsSize[i] > 1){
                                res.improvment = true;
                                //std::cout<<"push "<<anchors[i]<<"\n";
                                anchorQueue.push(anchors[i]);
                            }
                        }
                        
                        //std::cout<<"ret.minCutValue "<<res.minCutValue<<"\n";
                        //std::cout<<"ret.improvment "<<res.improvment<<"\n";
                    }   
                    return res;
                }
                else{   
                    return Optimzie1ReturnType(false,0.0);
                }
            }       

            template<class NODE_LABELS>
            Optimzie2ReturnType optimize2(
                NODE_LABELS & nodeLabels,
                const uint64_t anchorNode0, 
                const uint64_t anchorNode1
            ){
                // get mapping from local to global and vice versa
                // also counts nLocalEdges
                const auto maxNodeLabel = this->varMapping(nodeLabels, anchorNode0, anchorNode1);

                // setup the submodel
                SubGraph        subGraph(nLocalNodes_);
                MincutSubObjective    subObjective(subGraph);
                auto &  subWeights = subObjective.weights();

                ufd_.assign(nLocalNodes_);


                const auto anchorLabel0 = nodeLabels[anchorNode0];
                const auto anchorLabel1 = nodeLabels[anchorNode1];

                auto currentCutValue = 0.0;

                this->forEachInternalEdge(nodeLabels, anchorLabel0, anchorLabel1,[&](const uint64_t uLocal, const uint64_t vLocal, const uint64_t edge){
                    const auto w = weights_[edge];
                    const auto edgeId = subGraph.insertEdge(uLocal, vLocal);
                    subWeights.insertedEdges(edgeId);
                    subWeights[edgeId] = w;
                    if(nodeLabels[localNodeToGlobal_[uLocal]] != nodeLabels[localNodeToGlobal_[vLocal]]){
                        currentCutValue += w;
                    }
                });

                // optimize
                MincutSubNodeLabels subgraphRes(subGraph);
                auto solverPtr = mincutFactory_->create(subObjective);
                solverPtr->optimize(subgraphRes,nullptr);
                const auto minCutValue  = subObjective.evalNodeLabels(subgraphRes);
                delete solverPtr;   

                

                Optimzie2ReturnType ret(false, 0.0);

                // is there an improvement
                if(minCutValue + 1e-7 < currentCutValue){
                    

                    this->forEachInternalEdge(nodeLabels, anchorLabel0, anchorLabel1,[&](const uint64_t uLocal, const uint64_t vLocal, const uint64_t edge){
                        if(subgraphRes[uLocal] == subgraphRes[vLocal]){
                            ufd_.merge(uLocal, vLocal);
                        }
                    });


                    std::unordered_map<uint64_t,uint64_t> mapping;
                    ufd_.representativeLabeling(mapping);

                    for(auto localNode=0; localNode<nLocalNodes_; ++localNode){
                        const auto node = localNodeToGlobal_[localNode];
                        const auto denseLocalLabel =  mapping[ufd_.find(localNode)];
                        nodeLabels[node] = denseLocalLabel + maxNodeLabel + 1;
                    }
                    ret.improvment = true;
                    ret.improvedBy = currentCutValue - minCutValue;


                    // update isDirty 
                    if(ufd_.numberOfSets() <= 2){
                        // set inside to clean
                        // border to dirty
                        for(const auto edge : insideEdges_){
                            // already done
                            // isDirtyEdge_[edge] = false;
                        }
                        for(const auto edge : borderEdges_){
                            isDirtyEdge_[edge] = true;
                        }
                    }
                    else{
                        for(const auto edge : insideEdges_){
                            isDirtyEdge_[edge] = true;
                        }
                        for(const auto edge : borderEdges_){
                            isDirtyEdge_[edge] = true;
                        }
                    }
                
                }
                return ret;  
            }
        private:

            template<class NODE_LABELS>
            uint64_t varMapping(
                const NODE_LABELS & nodeLabels,
                const uint64_t anchorNode0
            ){
                return varMapping(nodeLabels, anchorNode0, anchorNode0);
            }

            template<class NODE_LABELS>
            uint64_t varMapping(
                 const NODE_LABELS & nodeLabels,
                const uint64_t anchorNode0, 
                const uint64_t anchorNode1
            ){

                insideEdges_.clear();
                borderEdges_.clear();

                nLocalNodes_ = 0;
                nLocalEdges_ = 0;
                uint64_t maxNodeLabel = 0;
                const auto anchorLabel0 = nodeLabels[anchorNode0];
                const auto anchorLabel1 = nodeLabels[anchorNode1];
                for(const auto node : graph_.nodes()){
                    const auto nodeLabel = nodeLabels[node];
                    maxNodeLabel = std::max(maxNodeLabel, nodeLabel);
                    if(nodeLabel == anchorLabel0 || nodeLabel == anchorLabel1){

                        globalNodeToLocal_[node] = nLocalNodes_;
                        localNodeToGlobal_[nLocalNodes_] = node;
                        nLocalNodes_ += 1;

                        for(const auto adj : graph_.adjacency(node)){
                            const auto otherNode = adj.node();
                            const auto edge = adj.edge();
                            if(node < otherNode){
                                const auto otherNodeLabel = nodeLabels[otherNode]; 
                                if(otherNodeLabel == anchorLabel0 || otherNodeLabel == anchorLabel1){
                                    nLocalEdges_ += 1;
                                    insideEdges_.push_back(edge);
                                    // mark inside edge as clear
                                    isDirtyEdge_[edge] = false;
                                }
                                // border node
                                else{
                                    borderEdges_.push_back(edge);
                                }
                            }
                        }
                    }
                }
                return maxNodeLabel;
            }

            template<class NODE_LABELS, class F>
            void forEachInternalEdge(const NODE_LABELS & nodeLabels, const uint64_t anchorLabel, F && f){
                for(auto localNode=0; localNode<nLocalNodes_; ++localNode){
                    const auto u = localNodeToGlobal_[localNode];
                    const auto uLocal = globalNodeToLocal_[u];
                    NIFTY_CHECK_OP(localNode,==,uLocal,"");
                    for(const auto adj : graph_.adjacency(u)){
                        const auto v = adj.node();
                        const auto edge = adj.edge();
                        if(u < v  && nodeLabels[v] == anchorLabel){
                            const auto vLocal = globalNodeToLocal_[v];
                            f(uLocal, vLocal, edge);
                        }
                    }
                }
            }

            template<class NODE_LABELS, class F>
            void forEachInternalEdge(
                const NODE_LABELS & nodeLabels, 
                const uint64_t anchorLabel0, 
                const uint64_t anchorLabel1,
                F && f
            ){
                for(auto localNode=0; localNode<nLocalNodes_; ++localNode){
                    const auto u = localNodeToGlobal_[localNode];
                    const auto uLocal = globalNodeToLocal_[u];
                    NIFTY_CHECK_OP(localNode,==,uLocal,"");
                    for(const auto adj : graph_.adjacency(u)){
                        const auto v = adj.node();
                        const auto edge = adj.edge();
                        if(u < v){
                            const auto vLabel = nodeLabels[v];
                            if(vLabel == anchorLabel0 || vLabel == anchorLabel1){
                                const auto vLocal = globalNodeToLocal_[v];
                                f(uLocal, vLocal, edge);
                            }
                        }
                    }
                }
            }


            const ObjectiveType & objective_; 
            const GraphType & graph_;
            const WeightsMapType & weights_;
            GlobalNodeToLocal globalNodeToLocal_;
            LocalNodeToGlobal localNodeToGlobal_;

            uint64_t nLocalNodes_;
            uint64_t nLocalEdges_;


            nifty::ufd::Ufd<uint64_t> ufd_;
            IsDirtyEdge & isDirtyEdge_;
            std::shared_ptr<MincutSubMcFactoryBase> &  mincutFactory_;

            std::vector<uint64_t> insideEdges_;
            std::vector<uint64_t> borderEdges_;
        };

        template<class OBJECTIVE>
        class Cgc ;

        template<class OBJECTIVE, class SETTINGS>
        class PartitionCallback{
        public:

            typedef SETTINGS SettingsType;

            typedef PartitionCallback<OBJECTIVE, SETTINGS> SelfType;
            typedef OBJECTIVE ObjectiveType;
            typedef typename ObjectiveType::Graph Graph;
            typedef typename ObjectiveType::GraphType GraphType;
            typedef typename GraphType:: template NodeMap<uint64_t> CcNodeSize;
            typedef typename GraphType:: template EdgeMap<float>    McWeights;
            typedef nifty::tools::ChangeablePriorityQueue< double ,std::less<double> > QueueType;


            PartitionCallback(
                const ObjectiveType & objective,
                const SettingsType & settings
            )
            :   objective_(objective),
                graph_(objective.graph()),
                pq_(objective.graph().edgeIdUpperBound()+1 ),
                ccNodeSize_(objective.graph()),
                mcWeights_(objective_.graph()),
                settings_(settings),
                currentNodeNum_(objective.graph().numberOfNodes()),
                edgeContractionGraph_(objective.graph(), *this)
            {
                //
                //std::cout<<"nodeNumStopCond "<<settings_.nodeNumStopCond<<"\n";
                //std::cout<<"sizeRegularizer "<<settings_.sizeRegularizer<<"\n";
                this->reset();
            }

            void reset(){

                // reset queue in case something is left
                while(!pq_.empty())
                    pq_.pop();
                for(const auto node: graph_.nodes()){
                    ccNodeSize_[node] = 1;
                }
                const auto & weights = objective_.weights();
                for(const auto edge: graph_.edges()){
                    mcWeights_[edge] =  weights[edge];
                    
                    pq_.push(edge, this->computeWeight(edge));
     
                }
                this->edgeContractionGraph_.reset();    
            }

            void contractEdge(const uint64_t edgeToContract){
                NIFTY_ASSERT(pq_.contains(edgeToContract));
                pq_.deleteItem(edgeToContract);
            }

            void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode){
                ccNodeSize_[aliveNode] += ccNodeSize_[deadNode];
                --currentNodeNum_;
            }

            void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge){
                NIFTY_ASSERT(pq_.contains(aliveEdge));
                NIFTY_ASSERT(pq_.contains(deadEdge));
                mcWeights_[aliveEdge] += mcWeights_[deadEdge];
                ///const auto wEdgeInAlive = pq_.priority(aliveEdge);
                //const auto wEdgeInDead = pq_.priority(deadEdge);
                pq_.deleteItem(deadEdge);
                //pq_.changePriority(aliveEdge, wEdgeInAlive + wEdgeInDead);
            }
            double computeWeight(const uint64_t edge)const{
                const auto su = double(ccNodeSize_[edgeContractionGraph_.u(edge)]);
                const auto sv = double(ccNodeSize_[edgeContractionGraph_.v(edge)]);
                const auto sr = settings_.sizeRegularizer;
                const auto sFac =2.0 / ( 1.0/std::pow(su, sr) + 1.0/std::pow(sv, sr) );
                const auto p1 =  1.0/(std::exp(mcWeights_[edge])+1.0);
                //std::cout<<"w "<<mcWeights_[edge]<<" p "<<p1<<" sFac "<<sFac<<"\n";
                return p1 * sFac;
            }
            void contractEdgeDone(const uint64_t edgeToContract){
                // HERE WE UPDATE 
                const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
                for(auto adj : edgeContractionGraph_.adjacency(u)){
                    const auto edge = adj.edge();
                    pq_.push(edge, computeWeight(edge));
                }
            }
            bool done(){

                const auto nnsc = settings_.nodeNumStopCond;
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
                if(currentNodeNum_<=1){
                    return true;
                }
                if(pq_.empty()){
                    return true;
                }
                return false;
            }

            uint64_t edgeToContract(){
                //
                return pq_.top();
            }

            void changeSettings(const SettingsType & settings){
                settings_ = settings;
            }
            const QueueType & queue()const{
                //
                return pq_;
            }
            EdgeContractionGraph<Graph, SelfType> & edgeContractionGraph(){
                return edgeContractionGraph_;
            }
            const EdgeContractionGraph<Graph, SelfType> & edgeContractionGraph()const{
                return edgeContractionGraph_;
            }
        private:

            const ObjectiveType & objective_;
            const GraphType & graph_;
            QueueType pq_;
            CcNodeSize ccNodeSize_;
            McWeights mcWeights_;
            SettingsType settings_;
            uint64_t currentNodeNum_;

            EdgeContractionGraph<GraphType, SelfType> edgeContractionGraph_;
        };

    }
    /// \endcond




    template<class OBJECTIVE>
    class Cgc : public MulticutBase<OBJECTIVE>
    {
    
    public: 

        typedef OBJECTIVE Objective;    
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<ObjectiveType> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::EdgeLabels EdgeLabels;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;

        typedef UndirectedGraph<>                              SubGraph;
        typedef mincut::MincutObjective<SubGraph, double>      MincutSubObjective;
        typedef mincut::MincutBase<MincutSubObjective>         MincutSubBase;
        typedef nifty::graph::optimization::common::SolverFactoryBase<MincutSubBase>  MincutSubMcFactoryBase;
        
        typedef typename  MincutSubBase::NodeLabels            MincutSubNodeLabels;

        typedef MulticutObjective<SubGraph, double>         MulticutSubObjective;
        typedef MulticutBase<MulticutSubObjective>          MulticutSubBase;
        typedef nifty::graph::optimization::common::SolverFactoryBase<MulticutSubBase>   MulticutSubMcFactoryBase;
       
        typedef typename  MulticutSubBase::NodeLabels       MulticutSubNodeLabels;


        typedef nifty::graph::optimization::common::SolverFactoryBase<BaseType> FactoryBase;
       
    private:
        typedef ComponentsUfd<Graph> Components;
    
    public:

        struct SettingsType {
            double nodeNumStopCond{0.1};
            double sizeRegularizer{1.0};
            bool doCutPhase{true};
            bool doBetterCutPhase{false};
            bool doGlueAndCutPhase{true};

            std::shared_ptr<MincutSubMcFactoryBase>   mincutFactory;
            std::shared_ptr<MulticutSubMcFactoryBase> multicutFactory;
        };
    private:
        typedef detail_cgc::PartitionCallback<OBJECTIVE, SettingsType> CallbackType;
        //typedef typename CallbackType::SettingsType      CallbackSettingsType;
    public:
        virtual ~Cgc(){
            
        }
        Cgc(const Objective & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBaseType * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("Cgc");
        }
        virtual void weightsChanged(){ 
        }
        virtual double currentBestEnergy() {
           return currentBestEnergy_;
        }
    private:


        void cutPhase(VisitorProxy & visitorProxy);
        void betterCutPhase(VisitorProxy & visitorProxy);

        void glueAndCutPhase(VisitorProxy & visitorProxy);

        const Objective & objective_;
        const Graph & graph_;
        const WeightsMap & weights_;

        Components components_;
        SettingsType settings_;
        IsDirtyEdge isDirtyEdge_;
        detail_cgc::SubmodelOptimizer<Objective> submodel_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;

        


    };

    
    template<class OBJECTIVE>
    Cgc<OBJECTIVE>::
    Cgc(
        const Objective & objective, 
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        weights_(objective_.weights()),
        components_(graph_),
        settings_(settings),
        isDirtyEdge_(graph_,true),
        submodel_(objective, isDirtyEdge_,  settings_.mincutFactory),
        currentBest_(nullptr),
        currentBestEnergy_(std::numeric_limits<double>::infinity())
    {

    }


    template<class OBJECTIVE>
    void Cgc<OBJECTIVE>::
    cutPhase(
        VisitorProxy & visitorProxy
    ){

       

        // the node labeling as reference
        auto & nodeLabels = *currentBest_;

        
        // number of components
        const auto nComponents = components_.buildFromLabels(nodeLabels);
        components_.denseRelabeling(nodeLabels);

        // get anchor for each component        
        std::vector<uint64_t> componentsAnchors(nComponents);

        // anchors
        graph_.forEachNode([&](const uint64_t node){
            componentsAnchors[nodeLabels[node]] = node;
        });

        std::queue<uint64_t> anchorQueue;
        for(const auto & anchor : componentsAnchors){
            anchorQueue.push(anchor);
        }


        // while nothing is on the queue
        visitorProxy.clearLogNames();
        visitorProxy.addLogNames({std::string("QueueSize"),std::string("E"),std::string("EE")});

        while(!anchorQueue.empty()){

            //std::cout<<"a) "<<anchorQueue.size()<<"\n";
            const auto anchorNode = anchorQueue.front();
            anchorQueue.pop();
            const auto anchorLabel = nodeLabels[anchorNode];
            //std::cout<<"b) "<<anchorQueue.size()<<"\n";
            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "Optimzie1");
            // optimize the submodel 
            const auto ret = submodel_.optimize1(nodeLabels, anchorNode, anchorQueue);
            if(ret.improvment){
                //std::cout<<"old current best "<<currentBestEnergy_<<"\n";
                currentBestEnergy_ += ret.minCutValue;
                //std::cout<<"new current best "<<currentBestEnergy_<<"\n";
            }
            //std::cout<<"c) "<<anchorQueue.size()<<"\n";
            visitorProxy.setLogValue(0, anchorQueue.size());        
            visitorProxy.setLogValue(1, currentBestEnergy_);    
            visitorProxy.setLogValue(2, objective_.evalNodeLabels(*currentBest_));   
            visitorProxy.visit(this);
        }

        visitorProxy.visit(this);



        visitorProxy.clearLogNames();
    }

    template<class OBJECTIVE>
    void Cgc<OBJECTIVE>::
    betterCutPhase(
        VisitorProxy & visitorProxy
    ){
        visitorProxy.clearLogNames();
        visitorProxy.addLogNames({std::string("#Node"),std::string("priority")});

        if(! bool(settings_.multicutFactory)){
            throw std::runtime_error("if betterCutPhase is used multicutFactory shall not be empty");
        }

        // the node labeling as reference
        auto & nodeLabels = *currentBest_;

        // run the cluster algorithm
        CallbackType callback(objective_, settings_);
        auto & edgeContractionGraph = callback.edgeContractionGraph(); 
        while(!callback.done() ){
            edgeContractionGraph.contractEdge(callback.edgeToContract());
            visitorProxy.setLogValue(0, edgeContractionGraph.numberOfNodes());
            visitorProxy.setLogValue(1, callback.queue().topPriority());
            if(!visitorProxy.visit(this))
                break;
        }
        


        // build a dense remapping
        const auto & nodeUfd = callback.edgeContractionGraph().nodeUfd();
        std::unordered_map<uint64_t,uint64_t> representativeLabeling;
        nodeUfd.representativeLabeling(representativeLabeling);

        auto ccIndex = [&](const uint64_t node){
            return representativeLabeling[nodeUfd.find(node)];
        };

        // submodels
        const auto nSubModels = nodeUfd.numberOfSets();


        std::vector<SubGraph *>              subGraphs(nSubModels, nullptr);
        std::vector<MulticutSubObjective *>  subObjectives(nSubModels, nullptr);
        std::vector<MulticutSubNodeLabels *> subLabels(nSubModels, nullptr);
        std::vector<uint64_t>                subModelSize(nSubModels, 0);



        visitorProxy.printLog(nifty::logging::LogLevel::INFO, std::string("nSubModels")
            + std::to_string(nSubModels));

        //std::cout<<"a\n";
        std::vector< std::unordered_map<uint64_t, uint64_t>  > nodeToSubNodeVec(nSubModels);
        for(const auto node : graph_.nodes()){
            const auto subgraphId = ccIndex(node);
            auto & nodeToSubNode = nodeToSubNodeVec[subgraphId];
            const auto subVarId = nodeToSubNode.size();
            nodeToSubNode[node] = subVarId;
        }

        //std::cout<<"b\n";
        // submodel size
        for(const auto node : graph_.nodes()){
            ++subModelSize[ccIndex(node)];
        }

        //std::cout<<"c\n";
        // allocs model and graphs
        for(auto i=0; i<nSubModels; ++i){
            subGraphs[i] = new SubGraph(subModelSize[i]);
            subObjectives[i] = new MulticutSubObjective(*subGraphs[i]);
            subLabels[i] = new MulticutSubNodeLabels(*subGraphs[i]);
        }

        //std::cout<<"d\n";
        // edges
        for(const auto edge : graph_.edges()){

            const auto u = graph_.u(edge);
            const auto v = graph_.v(edge);

            const auto ccU = ccIndex(u);
            const auto ccV = ccIndex(v);

            if(ccU == ccV){

                const auto & nodeToSubNode = nodeToSubNodeVec[ccU];

                const auto dU = nodeToSubNode.find(u)->second;
                const auto dV = nodeToSubNode.find(v)->second;
        
                //std::cout<<weights_.size()<<" ws\n";
                const auto w = weights_[edge];
                const auto edgeId = subGraphs[ccU]->insertEdge(dU, dV);
                subObjectives[ccU]->weights().insertedEdges(edgeId);
                subObjectives[ccU]->weights()[edgeId] = w;
                isDirtyEdge_[edge] = false;
            }
            else{
                isDirtyEdge_[edge] = true;
            }
        }   
        nifty::ufd::Ufd<uint64_t> ufd(graph_.nodeIdUpperBound()+1);


        visitorProxy.clearLogNames();
        visitorProxy.addLogNames({std::string("#subSize")});


        // allocs optimizers and optimize
        for(auto i=0; i<nSubModels; ++i){

            visitorProxy.setLogValue(0, subGraphs[i]->numberOfNodes());
            visitorProxy.visit(this);

            auto solver = settings_.multicutFactory->create(*subObjectives[i]);
            solver->optimize(*subLabels[i],nullptr);
            delete solver;  
        }
        //std::cout<<"merge\n";
        // merge
        for(const auto edge : graph_.edges()){

            const auto u = graph_.u(edge);
            const auto v = graph_.v(edge);

            const auto ccU = ccIndex(u);
            const auto ccV = ccIndex(v);

            if(ccU == ccV){

                const auto & nodeToSubNode = nodeToSubNodeVec[ccU];

                const auto dU = nodeToSubNode.find(u)->second;
                const auto dV = nodeToSubNode.find(v)->second;
        
                if((*subLabels[ccU])[dU] == (*subLabels[ccU])[dU]){
                    ufd.merge(u, v);
                }
            }
        }
        std::cout<<"eval\n";
        for(const auto node : graph_.nodes()){
            nodeLabels[node] = ufd.find(node);
        }
        currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);

        // de-allocs
        for(auto i=0; i<nSubModels; ++i){
            //delete subSolvers[i];
            delete subLabels[i];
            delete subObjectives[i];
            delete subGraphs[i];
        }

    }

    template<class OBJECTIVE>
    void Cgc<OBJECTIVE>::
    glueAndCutPhase(
        VisitorProxy & visitorProxy
    ){
        currentBestEnergy_ = objective_.evalNodeLabels(*currentBest_);

        visitorProxy.clearLogNames();
        visitorProxy.addLogNames({std::string("Sweep"),std::string("be")});

        // one anchor for all ``cc-edges``
        typedef std::pair<uint64_t, uint64_t> LabelPair;
        struct LabelPairHash{
        public:
            size_t operator()(const std::pair<uint64_t, uint64_t> & x) const {
                 size_t h = std::hash<uint64_t>()(x.first) ^ std::hash<uint64_t>()(x.second);
                 return h;
            }
        };
        typedef std::unordered_map<LabelPair, uint64_t, LabelPairHash> LabelPairToAnchor;
        LabelPairToAnchor labelPairToAnchorEdge;

        // the node labeling as reference
        auto & nodeLabels = *currentBest_;

        // initially everything is marked as dirty

        auto continueSeach = true;
        auto sweep = 0;
        while(continueSeach){   

            continueSeach = false;


            labelPairToAnchorEdge.clear();
            for(const auto  edge : graph_.edges()){
                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                const auto lu = nodeLabels[std::min(u,v)];
                const auto lv = nodeLabels[std::max(u,v)];
                if(lu != lv){
                    const auto labelPair = LabelPair(lu, lv);
                    labelPairToAnchorEdge.insert(std::make_pair(labelPair, edge));
                }
            }

            for(const auto kv : labelPairToAnchorEdge){
                const auto edge = kv.second;

                if(isDirtyEdge_[edge]){

                    const auto u = graph_.u(edge);
                    const auto v = graph_.v(edge);
                    const auto lu = nodeLabels[u];
                    const auto lv = nodeLabels[v];

                    // if the labels are still different
                    if(lu != lv){
                        const auto ret = submodel_.optimize2(nodeLabels, u, v);
                        if(ret.improvment){
                            continueSeach = true;
                            currentBestEnergy_ -= ret.improvedBy;

                            visitorProxy.setLogValue(0, sweep);
                            visitorProxy.setLogValue(1, currentBestEnergy_);                   
                            visitorProxy.visit(this);
                        }
                    }
                }
            }
            ++sweep;
        }
    }


    template<class OBJECTIVE>
    void Cgc<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBaseType * visitor
    ){  

        
        VisitorProxy visitorProxy(visitor);
        //visitorProxy.addLogNames({"violatedConstraints"});

        currentBest_ = &nodeLabels;
        currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);
        
        visitorProxy.begin(this);

        // the main workhorses
        // cut phase 
        if(settings_.doCutPhase){

            visitorProxy.printLog(nifty::logging::LogLevel::INFO, "Start Cut Phase:");
            if(settings_.doBetterCutPhase)
                this->betterCutPhase(visitorProxy);
            else
                this->cutPhase(visitorProxy);        
        }
        // glue phase
        if(settings_.doGlueAndCutPhase){
            visitorProxy.printLog(nifty::logging::LogLevel::INFO, "Start Glue & Cut Phase:");
            this->glueAndCutPhase(visitorProxy);
        }


        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename Cgc<OBJECTIVE>::Objective &
    Cgc<OBJECTIVE>::
    objective()const{
        return objective_;
    }



} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
