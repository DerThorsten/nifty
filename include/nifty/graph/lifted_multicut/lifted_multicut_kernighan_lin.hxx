#pragma once
#ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_Kernighan_LIN_HXX
#define NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_Kernighan_LIN_HXX


#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{




namespace detail_kernighang_lin{

    struct TwoCutSettings {
        std::size_t numberOfIterations { std::numeric_limits<std::size_t>::max() };
        double epsilon { 1e-9 };
    };

    template<class GRAPH>
    struct TwoCutBuffers{

        typedef GRAPH GraphType;
        
        TwoCutBuffers(const GraphType & graph)
        :   differences(graph),
            isMoved(graph),
            referencedBy(graph),
            vertexLabels(graph)
        {

        }

        typename GraphType:: template NodeMap<double>         differences;
        typename GraphType:: template NodeMap<char>           isMoved;
        typename GraphType:: template NodeMap<std::size_t>    referencedBy;
        typename GraphType:: template NodeMap<uint64_t>       vertexLabels;

        std::size_t maxNotUsedLabel;
    };

} // end namespace detail_kernighang_lin






    template<class OBJECTIVE>
    class LiftedMulticutKernighanLin : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<Objective> BaseType;
        typedef typename Objective::GraphType GraphType;
        typedef typename Objective::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::NodeLabels NodeLabels;



        typedef ComponentsUfd<GraphType> Components;
        typedef detail_kernighang_lin::TwoCutBuffers<GraphType> TwoCutBuffersType;

        struct GraphSubgraphWithCutFromNodeLabels {
            GraphSubgraphWithCutFromNodeLabels(
                const GraphType & graph,
                const NodeLabels & nodeLabels
            )
            :   graph_(graph),
                nodeLabels_(nodeLabels){
            }

            bool useNode(const uint64_t v) const{ 
                return true; 
            }
            bool useEdge(const uint64_t graphEdge)const{ 
                const auto uv = graph_.uv(graphEdge);
                return nodeLabels_[uv.first] != nodeLabels_[uv.second];
            }
            const GraphType & graph_;
            const NodeLabels & nodeLabels_;
        };

    public:

        

        struct Settings{

            std::size_t numberOfInnerIterations { std::numeric_limits<std::size_t>::max() };
            std::size_t numberOfOuterIterations { 100 };
            double epsilon { 1e-7 };
        };


        virtual ~LiftedMulticutKernighanLin(){}
        LiftedMulticutKernighanLin(const Objective & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;





 


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("LiftedMulticutKernighanLin");
        }


    private:




        const Objective & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabels * currentBest_;

        Components components_;
        std::vector< std::vector<uint64_t> > partitions_;
        TwoCutBuffersType twoCutBuffers_;
        NodeLabels lastGoodVertexLabels_;

        // auxillary array for BFS/DFS
        typename GraphType:: template NodeMap<uint8_t>  visited_;

        // 1 if i-th partitioned changed since last iteration, 0 otherwise
        std::vector<uint8_t>  changed_;

        //edges from labels connected component / rag graph
        std::vector<std::unordered_set<uint64_t> > edges_;
    };

    
    template<class OBJECTIVE>
    LiftedMulticutKernighanLin<OBJECTIVE>::
    LiftedMulticutKernighanLin(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.graph()),
        currentBest_(nullptr),
        //
        components_(objective.graph()),
        partitions_(),
        twoCutBuffers_(objective.graph()),
        lastGoodVertexLabels_(objective_.graph()),
        visited_(objective.graph()),
        changed_(),
        edges_()
    {

    }

    template<class OBJECTIVE>
    void LiftedMulticutKernighanLin<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);

        // components from current best node labels
        components_.build(GraphSubgraphWithCutFromNodeLabels(graph_, *currentBest_));



        // compute energy
        auto startingEnergy = objective_.evalNodeLabels(components_);

        // get maximum component index
        auto maxComponentLabel = components_.maxLabel();


        // build the explicit partitions
        // and remember the last good/valid vertex labels
        partitions_.resize(maxComponentLabel + 1);
        graph_.forEachNode([&](const uint64_t node){
            const auto ccLabel = components_[node];
            partitions_[ccLabel].push_back(node);
            twoCutBuffers_.vertexLabels[node] = ccLabel;
            lastGoodVertexLabels_[node] =  ccLabel;
        });
        twoCutBuffers_.maxNotUsedLabel = partitions_.size();


        changed_.resize(maxComponentLabel + 1);

        // interatively update bipartition in order to minimize the total cost of the multicut
        for (size_t k = 0; k < settings_.numberOfOuterIterations; ++k){

            auto energyDecrease = 0.0;

            // build components rag graph (just via edges)
            edges_.resize(maxComponentLabel+1);
            graph_.forEachEdge([&](const uint64_t edge){
                const auto uv = graph_.uv(edge);
                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    const auto lU = twoCutBuffers_.vertexLabels[uv.first];
                    const auto lV = twoCutBuffers_.vertexLabels[uv.second];
                    edges_[std::min(lU,lV)].insert(std::max(lU,lV));
                }
            });

            for(uint64_t piU=0; piU<maxComponentLabel+1; ++piU)
                if(!partitions_[piU].empty())
                    for(const auto piV : edges_[piU])
                        if (!partitions_[piV].empty() && (changed_[piU] || changed_[piV])){

                            // HERE WE TRY TO UPDATE THE PAIR OF PARTITIONS

                            auto ret = 0.0; // CALL TWOCUT HERE TODO

                            if(ret > settings_.epsilon){
                                changed_[piU] = 1;
                                changed_[piV] = 1;
                            }

                            energyDecrease += ret;

                            if(partitions_[piU].size() == 0)
                                break;
                            
                        }

            auto ee = energyDecrease;

            // remove partitions that became empty after the previous step
            auto partionF =  [](const std::vector<size_t>& s) { return !s.empty(); };
            auto new_end = std::partition(partitions_.begin(), partitions_.end(), partionF);
            partitions_.resize(new_end - partitions_.begin());




        }




        //if(!visitorProxy.visit(this))
        //    for;
        visitorProxy.end(this);     
    }

    template<class OBJECTIVE>
    const typename LiftedMulticutKernighanLin<OBJECTIVE>::Objective &
    LiftedMulticutKernighanLin<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 


    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_Kernighan_LIN_HXX
