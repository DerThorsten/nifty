// reimplementation of kerninhanlin in 
// https://github.com/bjoern-andres/graph

#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX


#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "vigra/priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/subgraph_masks/subgraph_with_cut.hxx"

#include "nifty/graph/optimization/lifted_multicut/detail/lifted_twocut_kernighan_lin.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{




    template<class OBJECTIVE>
    class LiftedMulticutKernighanLin : public LiftedMulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        
        typedef typename BaseType::VisitorBase VisitorBase;
        typedef typename BaseType::VisitorProxy VisitorProxy;
        typedef typename BaseType::NodeLabels NodeLabels;

    private:

        typedef detail_kernighang_lin::TwoCut<ObjectiveType> TwoCutType;
        typedef ComponentsUfd<GraphType> ComponentsType;
        typedef detail_kernighang_lin::TwoCutBuffers<GraphType> TwoCutBuffersType;

        typedef subgraph_masks::SubgraphWithCutFromNodeLabels<GraphType, NodeLabels> SubgraphWithCut;

    public:

        

        struct Settings{

            std::size_t numberOfInnerIterations { std::numeric_limits<std::size_t>::max() };
            std::size_t numberOfOuterIterations { 100 };
            double epsilon { 1e-7 };
        };


        virtual ~LiftedMulticutKernighanLin(){}
        LiftedMulticutKernighanLin(const ObjectiveType & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const ObjectiveType & objective() const;



        virtual const NodeLabels & currentBestNodeLabels( );
        virtual std::string name()const;
        virtual double currentBestEnergy();

    private:

        void initializePartiton();
        void buildRegionAdjacencyGraph();
        void optimizePairs(double & energyDecrease);

        void introduceNewPartitions(double & energyDecrease);
        void connectedComponentLabeling();
        bool hasChanges();

        void formNewPartition();

        void checkIfPartitonChanged();


        const ObjectiveType & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;

        TwoCutType twoCut_;
        ComponentsType components_;
        std::vector< std::vector<uint64_t> > partitions_;
        TwoCutBuffersType twoCutBuffers_;
        NodeLabels lastGoodVertexLabels_;

        // auxillary array for BFS/DFS
        typename GraphType:: template NodeMap<uint8_t>  visited_;

        // 1 if i-th partitioned changed since last iteration, 0 otherwise
        std::vector<uint8_t>  changed_;

        //edges from labels connected component / rag graph
        std::vector<std::unordered_set<uint64_t> > edges_;

        uint64_t numberOfComponents_;
    };

    
    template<class OBJECTIVE>
    LiftedMulticutKernighanLin<OBJECTIVE>::
    LiftedMulticutKernighanLin(
        const ObjectiveType & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        graph_(objective.graph()),
        liftedGraph_(objective.liftedGraph()),
        currentBest_(nullptr),
        currentBestEnergy_(0.0),
        //
        twoCut_(objective),
        components_(objective.graph()),
        partitions_(),
        twoCutBuffers_(objective.graph()),
        lastGoodVertexLabels_(objective.graph()),
        visited_(objective.graph()),
        changed_(),
        edges_(),
        numberOfComponents_(0)
    {

    }

    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        
        currentBest_ = &nodeLabels;

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);

        // build cc on given nodeLabels
        // and  build initial partiton
        this->initializePartiton();


        for(const auto node : graph_.nodes()){
            visited_[node] = 0;
        }

        // interatively update bipartition in order to minimize the total cost of the multicut
        for (size_t k = 0; k < settings_.numberOfOuterIterations; ++k){


            auto energyDecrease = 0.0;

            this->buildRegionAdjacencyGraph();
            this->optimizePairs(energyDecrease);
            this->introduceNewPartitions(energyDecrease);

            if(energyDecrease == .0)
                break;

            this->connectedComponentLabeling();
            double newEnergyValue = objective_.evalNodeLabels(twoCutBuffers_.referencedBy);


            // if the new true energy is higher, than the current one, revert the changes and terminate
            if (newEnergyValue >= currentBestEnergy_ - settings_.epsilon){
                for(const auto node : graph_.nodes())
                    nodeLabels[node] = lastGoodVertexLabels_[node];
                break;
            }

            // otherwise, form new partitions
            this->formNewPartition();

            for (const auto i : graph_.nodes()){
                nodeLabels[i] = twoCutBuffers_.vertexLabels[i];
            }

            // check if labeling changed
            if(!hasChanges()){
                break;
            }

            // check if the shape of some partitions didn't change
            this->checkIfPartitonChanged();

            //  remeber last labeling
            for(const auto  node : graph_.nodes())
                lastGoodVertexLabels_[node] = twoCutBuffers_.vertexLabels[node];

            // update best energy
            currentBestEnergy_ = newEnergyValue;

            // call visitor
            if(!visitorProxy.visit(this))
                break;

        }
        visitorProxy.end(this);     
    }


    

    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    initializePartiton(
    ){

        // components from current best node labels
        components_.build(SubgraphWithCut(graph_, *currentBest_));

        // compute energy
        currentBestEnergy_ = objective_.evalNodeLabels(components_);

        // get maximum component index
        numberOfComponents_ = components_.maxLabel() + 1;


        // build the explicit partitions
        // and remember the last good/valid vertex labels\
    
        partitions_.clear();
        partitions_.resize(numberOfComponents_);

        graph_.forEachNode([&](const uint64_t node){
            const auto ccLabel = components_[node];
            (*currentBest_)[node] = ccLabel;
            partitions_[ccLabel].push_back(node);
            twoCutBuffers_.vertexLabels[node] = ccLabel;
            lastGoodVertexLabels_[node] =  ccLabel;
        });
        twoCutBuffers_.maxNotUsedLabel = partitions_.size();

        std::fill(changed_.begin(), changed_.end(),1);
        changed_.resize(numberOfComponents_, 1);
        
    }


    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    buildRegionAdjacencyGraph(
    ){
        edges_.clear();
        edges_.resize(numberOfComponents_);
        graph_.forEachEdge([&](const uint64_t edge){
            const auto uv = graph_.uv(edge);
            
            const auto lU = twoCutBuffers_.vertexLabels[uv.first];
            const auto lV = twoCutBuffers_.vertexLabels[uv.second];

            if(lU != lV){
                edges_[std::min(lU,lV)].insert(std::max(lU,lV));
            }
            
        });
    }


    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    optimizePairs(
        double & energyDecrease
    ){
        for(uint64_t piU=0; piU<numberOfComponents_; ++piU){
            auto & pU = partitions_[piU];
            if(!pU.empty()){
                for(const auto piV : edges_[piU]){
                    auto & pV = partitions_[piV];
                    if (!pV.empty() && (changed_[piU] || changed_[piV])){

                        // HERE WE TRY TO UPDATE THE PAIR OF PARTITIONS
                        const auto ret = twoCut_.optimizeTwoCut(pU, pV, twoCutBuffers_);
                        //std::cout<<"to cut ret "<<ret<<"\n";
                        if(ret > settings_.epsilon){
                            changed_[piU] = 1;
                            changed_[piV] = 1;
                        }

                        energyDecrease += ret;

                        if(partitions_[piU].size() == 0)
                            break;
                        
                    }
                }
            }
        }

        // remove partitions that became empty after the previous step
        auto partionF =  [](const std::vector<uint64_t>& s) { return !s.empty(); };
        auto newEnd = std::partition(partitions_.begin(), partitions_.end(), partionF);
        partitions_.resize(newEnd - partitions_.begin());

    }


    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    introduceNewPartitions(
        double & energyDecrease
    ){
        for (std::size_t i = 0, pSize = partitions_.size(); i < pSize; ++i){

            // CALL VISITOR HERE

            if (!changed_[i])
                continue;

            bool flag = true;
            while (flag){

                // CALL VISITOR HERE
                std::vector<uint64_t> newSet;
                energyDecrease += twoCut_.optimizeTwoCut(partitions_[i], newSet, twoCutBuffers_);

                flag = !newSet.empty();
                if (!newSet.empty())
                    partitions_.emplace_back(std::move(newSet));
            }
        }
    }

    
    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    connectedComponentLabeling(){

        std::stack<std::size_t> S;
        for (const auto node : graph_.nodes()){
            visited_[node] = 0;
        }

        // do connected component labeling on the original graph
        numberOfComponents_ = 0;

        for (const auto i : graph_.nodes()){
            if (!visited_[i]){

                S.push(i);
                visited_[i] = 1;

                const auto label = twoCutBuffers_.vertexLabels[i];

                twoCutBuffers_.referencedBy[i] = numberOfComponents_;

                while (!S.empty()){

                    auto v = S.top();
                    S.pop();

                    for(const auto adj : graph_.adjacency(v)){

                        if (twoCutBuffers_.vertexLabels[adj.node()] == label && !visited_[adj.node()]){
                            S.push(adj.node());
                            visited_[adj.node()] = 1;
                            twoCutBuffers_.referencedBy[adj.node()] = numberOfComponents_;
                        }
                    }
                }

                ++numberOfComponents_;
            }
        }  
    }


    template<class OBJECTIVE>
    inline bool 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    hasChanges(){

        bool didntChange = true;
        for (const auto i : graph_.edges()){
            const auto uv = graph_.uv(i);
            const auto v0 = uv.first;
            const auto v1 = uv.second;
            const auto edgeLabel  = (twoCutBuffers_.vertexLabels[v0] == twoCutBuffers_.vertexLabels[v1]) ? 0 : 1;

            if (static_cast<bool>(edgeLabel) != (lastGoodVertexLabels_[v0] != lastGoodVertexLabels_[v1]))
                didntChange = false;
        }
        return !didntChange;
    }

    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    formNewPartition(){
        partitions_.clear();
        partitions_.resize(numberOfComponents_);

        for (const auto i : graph_.nodes()){
            twoCutBuffers_.vertexLabels[i] = twoCutBuffers_.referencedBy[i];
            partitions_[twoCutBuffers_.vertexLabels[i]].push_back(i);
        }
        twoCutBuffers_.maxNotUsedLabel = numberOfComponents_;
    }


    


    template<class OBJECTIVE>
    inline void 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    checkIfPartitonChanged(){

        changed_.clear();
        changed_.resize(numberOfComponents_);
        std::fill(changed_.begin(), changed_.end(), 0);

        for (const auto node : graph_.nodes()){
            visited_[node] = 0;
        }

        std::stack<std::size_t> S;
        for (const auto i : graph_.nodes()){
            if (!visited_[i]){

                S.push(i);
                visited_[i] = 1;

                const auto labelNew = twoCutBuffers_.vertexLabels[i];
                const auto labelOld = lastGoodVertexLabels_[i];

                while (!S.empty()){

                    const auto v = S.top();
                    S.pop();


                    for(const auto adj : graph_.adjacency(v)){

      
                        if (lastGoodVertexLabels_[adj.node()] == labelOld && twoCutBuffers_.vertexLabels[adj.node()] != labelNew)
                            changed_[labelNew] = 1;

                        if (visited_[adj.node()])
                            continue;

                        if (twoCutBuffers_.vertexLabels[adj.node()] == labelNew)
                        {
                            S.push(adj.node());
                            visited_[adj.node()] = 1;

                            if (lastGoodVertexLabels_[adj.node()] != labelOld)
                                changed_[labelNew] = 1;
                        }
                    }
                }
            }
        }
    }


    template<class OBJECTIVE>
    const typename LiftedMulticutKernighanLin<OBJECTIVE>::ObjectiveType &
    LiftedMulticutKernighanLin<OBJECTIVE>::
    objective()const{
        return objective_;
    }

 
    template<class OBJECTIVE>
    inline 
    const typename LiftedMulticutKernighanLin<OBJECTIVE>::NodeLabels & 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    currentBestNodeLabels( ){
        return *currentBest_;
    }

    template<class OBJECTIVE>
    inline 
    std::string 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    name()const{
        return std::string("LiftedMulticutKernighanLin");
    }

    template<class OBJECTIVE>
    inline 
    double 
    LiftedMulticutKernighanLin<OBJECTIVE>::
    currentBestEnergy()  {
        return currentBestEnergy_;
    }



    
} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX
