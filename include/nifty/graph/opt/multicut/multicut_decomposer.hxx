#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/graph/opt/multicut/multicut_base.hxx"
#include "nifty/graph/opt/common/solver_factory.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    class MulticutDecomposer : public MulticutBase<OBJECTIVE>
    {
    public:

        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<OBJECTIVE> BaseType;
        typedef typename BaseType::VisitorBaseType VisitorBaseType;
        typedef typename BaseType::VisitorProxyType VisitorProxyType;
        typedef typename BaseType::NodeLabelsType NodeLabelsType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;


        typedef nifty::graph::opt::common::SolverFactoryBase<BaseType> FactoryBase;

    private:
        typedef ComponentsUfd<GraphType> Components;


    public:
        typedef UndirectedGraph<>                                                               SubmodelGraph;
        typedef MulticutObjective<SubmodelGraph, WeightType>                                    SubmodelObjective;
        typedef MulticutBase<SubmodelObjective>                                                 SubmodelMulticutBaseType;
        typedef nifty::graph::opt::common::SolverFactoryBase<SubmodelMulticutBaseType> SubmodelFactoryBase;

        typedef typename SubmodelMulticutBaseType::NodeLabelsType    SubmodelNodeLabels;

    public:

        struct SettingsType{
            std::shared_ptr<SubmodelFactoryBase> submodelFactory;
            std::shared_ptr<FactoryBase>         fallthroughFactory;
            int numberOfThreads;

        };

        virtual ~MulticutDecomposer(){

        }
        MulticutDecomposer(const ObjectiveType & objective, const SettingsType & settings = SettingsType());


        virtual void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor);
        virtual const ObjectiveType & objective() const;


        virtual const NodeLabelsType & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("MulticutDecomposer");
        }
        virtual void weightsChanged(){
        }

    private:

        struct SubgraphWithCut {
            SubgraphWithCut(const WeightsMap & weights)
                :   weights_(weights)
            {}
            bool useNode(const std::size_t v) const
                { return true; }
            bool useEdge(const std::size_t e) const
            {
                return weights_[e] > 0.0;
            }

            const WeightsMap & weights_;
        };





        const ObjectiveType & objective_;
        const GraphType & graph_;
        const WeightsMap & weights_;

        Components components_;
        NodeLabelsType * currentBest_;

        SettingsType settings_;
    };


    template<class OBJECTIVE>
    MulticutDecomposer<OBJECTIVE>::
    MulticutDecomposer(
        const ObjectiveType & objective,
        const SettingsType & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        weights_(objective_.weights()),
        components_(graph_),
        settings_(settings)
    {
        if(!bool(settings_.fallthroughFactory)){
            throw std::runtime_error("MulticutDecomposer SettingsType: fallthroughFactory may not be empty!");
        }
        if(!bool(settings_.submodelFactory)){
            throw std::runtime_error("MulticutDecomposer SettingsType: submodelFactory may not be empty!");
        }
    }

    template<class OBJECTIVE>
    void MulticutDecomposer<OBJECTIVE>::
    optimize(
        NodeLabelsType & nodeLabels,  VisitorBaseType * visitor
    ){


        VisitorProxyType visitorProxy(visitor);
        //visitorProxy.addLogNames({"violatedConstraints"});
        currentBest_ = &nodeLabels;

        visitorProxy.begin(this);


        // build the connected components
        NodeLabelsType denseLabels(graph_);
        const auto nComponents = components_.build(SubgraphWithCut(weights_));
        std::vector<std::size_t> componentsSize(nComponents,0);
        components_.denseRelabeling(denseLabels, componentsSize);


        visitorProxy.printLog(nifty::logging::LogLevel::INFO,
            std::string("model decomposes in ")+std::to_string(nComponents));


        // build the sub objectives in the case
        // the thing decomposes
        if(nComponents >= 2){

            visitorProxy.clearLogNames();
            visitorProxy.addLogNames({std::string("modelSize")});

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "alloc subgraphs");
            // first pass :
            // - allocate the sub graphs
            // - sparse to dense
            std::vector<SubmodelGraph *> subGraphVec(nComponents);
            for(std::size_t i=0; i<nComponents; ++i){
                NIFTY_CHECK_OP(componentsSize[i],>,0,"");
                if(componentsSize[i]>1)
                    subGraphVec[i] = new SubmodelGraph(componentsSize[i]);
            }


            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "global to local mappin");
            // map from global variables to
            // subproblem variables
            std::vector< std::unordered_map<uint64_t, uint64_t>  > nodeToSubNodeVec(nComponents);
            for(const auto node : graph_.nodes()){
                const auto subgraphId = denseLabels[node];
                auto & nodeToSubNode = nodeToSubNodeVec[subgraphId];
                const auto subVarId = nodeToSubNode.size();
                nodeToSubNode[node] = subVarId;
            }


            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "add edges");
            // add edges to subproblems
            for(const auto edge : graph_.edges()){

                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                const auto lu = denseLabels[u];

                if(componentsSize[lu]>1){
                    if(lu == denseLabels[v]){

                        NIFTY_CHECK_OP(lu,<,nComponents,"");

                        const auto & nodeToSubNode = nodeToSubNodeVec[lu];

                        const auto findU = nodeToSubNode.find(u);
                        const auto findV = nodeToSubNode.find(v);

                        NIFTY_CHECK(findU != nodeToSubNode.end(),"");
                        NIFTY_CHECK(findV != nodeToSubNode.end(),"");

                        const auto su = findU->second;
                        const auto sv = findV->second;

                        NIFTY_CHECK_OP(su,<,subGraphVec[lu]->numberOfNodes(),"");
                        NIFTY_CHECK_OP(sv,<,subGraphVec[lu]->numberOfNodes(),"");

                        subGraphVec[lu]->insertEdge(su,sv);
                    }
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "build sub-objectives");
            // build the sub mc objectives
            std::vector<SubmodelObjective *> subObjectiveVec(nComponents);
            for(std::size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1)
                    subObjectiveVec[i] = new SubmodelObjective(*subGraphVec[i]);
            }

            for(const auto edge : graph_.edges()){
                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                const auto lu = denseLabels[u];
                if(componentsSize[lu]>1){
                    if(lu == denseLabels[v]){
                        const auto & nodeToSubNode = nodeToSubNodeVec[lu];
                        const auto su = nodeToSubNode.find(u)->second;
                        const auto sv = nodeToSubNode.find(v)->second;
                        const auto subEdge = subGraphVec[lu]->findEdge(su,sv);
                        subObjectiveVec[lu]->weights()[subEdge] = weights_[edge];
                    }
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "optimize subproblems");
            // optimize the subproblems
            // and delete stuff we do not need anymore
            std::vector<SubmodelNodeLabels *> subNodeLabelsVec(nComponents);

            // //////////////////////////////////////////////
            // solving and partial cleanup
            // //////////////////////////////////////////////
            for(std::size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1){
                    const auto & subObj = *subObjectiveVec[i];
                    const auto & subGraph = *subGraphVec[i];
                    const auto nSubGraphNodes = subGraph.numberOfNodes();
                    // TODO log with visitor
                    // std::cout<<"#Nodes "<<nSubGraphNodes<<" "<<float(nSubGraphNodes)/float(graph_.numberOfNodes())<<"\n";

                    subNodeLabelsVec[i] = new SubmodelNodeLabels(subGraph);

                    // create solver and optimize
                    auto subSolver = settings_.submodelFactory->create(subObj);
                    subSolver->optimize(*subNodeLabelsVec[i], nullptr);
                    delete subSolver;
                    delete subObjectiveVec[i];
                    delete subGraphVec[i];
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "map sub to global");
            // map from the sub solutions to global solution
            {
                nifty::ufd::Ufd< > ufd(graph_.nodeIdUpperBound()+1);
                for(const auto edge : graph_.edges()){

                    const auto u = graph_.u(edge);
                    const auto v = graph_.v(edge);
                    const auto lu = denseLabels[u];

                    if(componentsSize[lu]>1){
                        if(lu == denseLabels[v]){

                            const auto & nodeToSubNode = nodeToSubNodeVec[lu];
                            const auto & subNodeLabels = *subNodeLabelsVec[lu];
                            const auto su = nodeToSubNode.find(u)->second;
                            const auto sv = nodeToSubNode.find(v)->second;
                            if(subNodeLabels[su] == subNodeLabels[sv]){
                                ufd.merge(u, v);
                            }
                        }
                    }
                }

                for(const auto node : graph_.nodes()){
                    nodeLabels[node] = ufd.find(node);
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "cleanup");
            // final cleanup
            for(std::size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1){
                    delete subNodeLabelsVec[i];
                }
            }
            visitorProxy.clearLogNames();
        }
        else{
            auto solverPtr = settings_.fallthroughFactory->create(objective_);
            // TODO handle visitor:
            // Problem: if we just pass the
            // visitor begin and end are called twice
            solverPtr->optimize(nodeLabels, nullptr);
            delete solverPtr;
        }

        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename MulticutDecomposer<OBJECTIVE>::ObjectiveType &
    MulticutDecomposer<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

