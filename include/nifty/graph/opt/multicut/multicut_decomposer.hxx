#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/parallel/threadpool.hxx"
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
        std::vector<std::size_t> componentsSize(nComponents, 0);
        components_.denseRelabeling(denseLabels, componentsSize);


        visitorProxy.printLog(nifty::logging::LogLevel::INFO,
            std::string("model decomposes in ")+std::to_string(nComponents));


        // build the sub objectives in the case
        // the thing decomposes
        if(nComponents >= 2){

            visitorProxy.clearLogNames();
            visitorProxy.addLogNames({std::string("modelSize")});

            // get list of global node ids for all components
            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "global to local mappin");
            std::unordered_map<uint64_t, std::vector<uint64_t>> componentToGlobalNodes;
            for(const auto node : graph_.nodes()){
                const auto componentId = denseLabels[node];
                auto componentNodes = componentToGlobalNodes.find(componentId);
                if(componentNodes == componentToGlobalNodes.end()) {
                    componentToGlobalNodes[componentId] = std::vector<uint64_t>({static_cast<uint64_t>(node)});
                } else {
                    componentNodes->second.push_back(node);
                }
            }

            // create and solve the subproblems in parallel
            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "optimize subproblems");

            std::vector<SubmodelNodeLabels> subNodeLabelsVec(nComponents);
            nifty::parallel::ParallelOptions pOpts(settings_.numberOfThreads);
            nifty::parallel::ThreadPool threadPool(pOpts);

            // sort the component ids by size in descending order, so that we start solving the larger problems first
            // to work on the critical path first
            std::vector<std::size_t> componentIdsSorted(nComponents);
            std::iota(componentIdsSorted.begin(), componentIdsSorted.end(), 0);
            std::sort(componentIdsSorted.begin(), componentIdsSorted.end(),
                      [&componentsSize](std::size_t cA, std::size_t cB){return componentsSize[cA] > componentsSize[cB];});

            nifty::parallel::parallel_foreach(threadPool, nComponents, [&](const std::size_t threadId, const std::size_t i){
                const std::size_t componentId = componentIdsSorted[i];
                if(componentsSize[componentId] > 1){
                    const auto & subNodes = componentToGlobalNodes.at(componentId);

                    // dense relabeling for the nodes in our subproblem
                    std::unordered_map<uint64_t, uint64_t> subRelabeling;
                    uint64_t localNodeId = 0;
                    for(auto nodeId: subNodes) {
                        subRelabeling[nodeId] = localNodeId;
                        ++localNodeId;
                    }

                    std::vector<int64_t> subEdgeIds;
                    graph_.edgesFromNodeList(subNodes, subEdgeIds);

                    // NOTE this should be faster, but appears not to be in practice
                    // copy to unordered set to find nodes faster
                    // std::unordered_set<uint64_t> subNodeSet(subNodes.begin(), subNodes.end());
                    // graph_.edgesFromNodeList(subNodeSet, subEdgeIds);

                    const uint64_t nSubNodes = subNodes.size();
                    SubmodelGraph subGraph(nSubNodes);
                    for(auto edgeId: subEdgeIds) {
                        const auto & uv = graph_.uv(edgeId);
                        subGraph.insertEdge(subRelabeling[uv.first], subRelabeling[uv.second]);
                    }

                    // std::cout<<"#Nodes "<<nSubGraphNodes<<" "<<float(nSubGraphNodes)/float(graph_.numberOfNodes())<<"\n";

                    SubmodelObjective subObj(subGraph);
                    auto & subWeights = subObj.weights();
                    int64_t localEdgeId = 0;
                    for(auto edgeId: subEdgeIds) {
                        subWeights[localEdgeId] = weights_[edgeId];
                        ++localEdgeId;
                    }

                    // create solver and optimize
                    SubmodelNodeLabels subLabels = SubmodelNodeLabels(subGraph);
                    auto subSolver = settings_.submodelFactory->create(subObj);
                    subSolver->optimize(subLabels, nullptr);

                    // dense relabeling of the results
                    uint64_t denseLabelId = 0;
                    std::unordered_map<uint64_t, uint64_t> resRelabeling;
                    for(std::size_t labelId = 0; labelId < subLabels.size(); ++labelId) {
                        uint64_t & nodeLabel = subLabels[labelId];
                        auto relabelIt = resRelabeling.find(nodeLabel);
                        if(relabelIt == resRelabeling.end()) {
                            resRelabeling[nodeLabel] = denseLabelId;
                            nodeLabel = denseLabelId;
                            ++denseLabelId;
                        } else {
                            nodeLabel = relabelIt->second;
                        }
                    }
                    subNodeLabelsVec[componentId] = subLabels;
                } else {
                    SubmodelGraph subGraph(1);
                    SubmodelNodeLabels subLabels(subGraph);
                    subLabels[0] = 0;
                    subNodeLabelsVec[componentId] = subLabels;
                }
            });

            // map from the sub solutions to global solution
            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "map sub to global");
            uint64_t labelOffset = 0;
            for(std::size_t i = 0; i < nComponents; ++i) {
                const auto & subNodes = componentToGlobalNodes.at(i);
                const auto & subLabels = subNodeLabelsVec[i];
                for(std::size_t j = 0; j < subNodes.size(); ++j) {
                    nodeLabels[subNodes[j]] = subLabels[j] + labelOffset;
                }
                labelOffset += (*std::max_element(subLabels.begin(), subLabels.end()) + 1);
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

