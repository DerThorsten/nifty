// reimplementation of kerninhanlin in 
// https://github.com/bjoern-andres/graph

#pragma once
#ifndef NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX
#define NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX


#include <iomanip>
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


    template<class OBJECTIVE>
    class TwoCut{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> BaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;



        struct Settings {
            std::size_t numberOfIterations { std::numeric_limits<std::size_t>::max() };
            double epsilon { 1e-9 };
        };

        struct Move{
            int64_t v { -1 };
            double difference { std::numeric_limits<double>::lowest() };
            uint64_t newLabel;

            void updateDifference(const uint64_t var, const double diff){
                if(diff > difference){
                    //std::cout<<"update diff from  "<<difference<<" to "<<diff<<"\n";
                    v = var;
                    difference = diff;
                }
            }
        };

        TwoCut(
            const ObjectiveType & objective,
            const Settings & settings = Settings()
        )
        :   objective_(objective),
            settings_(settings),
            graph_(objective.graph()),
            liftedGraph_(objective.liftedGraph()),
            weights_(objective.weights()),
            border_(),
            moves_(){
        }

        template<class SET>
        double optimizeTwoCut(
            SET & A,
            SET & B,
            TwoCutBuffers<GraphType> & buffer
        ){
            auto gainFromMerging = 0.0;
            auto computeDifferences = [&](const SET & varSet, const uint64_t labelAA, const uint64_t labelBB){
                for(const auto var : varSet){

                    double diffExt = 0.0;
                    double diffInt = 0.0;
                    uint64_t refCnt = 0;

                    for(auto adj : liftedGraph_.adjacency(var)){

                        const auto label =  buffer.vertexLabels[adj.node()];
                        const auto edge = adj.edge();

                        if (label == labelAA){
                            //std::cout<<"e "<<edge<<" w "<<weights_[edge]<<"\n";
                            diffInt += weights_[edge];
                        }
                        else if (label == labelBB){
                            //std::cout<<"e "<<edge<<" w "<<weights_[edge]<<"\n";
                            diffExt += weights_[edge];
                        }
                    }

                    for(auto adj : graph_.adjacency(var)){
                        if (buffer.vertexLabels[adj.node()] == labelBB)
                            ++refCnt;
                    }

                    buffer.differences[var] = diffExt - diffInt;
                    buffer.referencedBy[var] = refCnt;
                    buffer.isMoved[var] = 0;

                    gainFromMerging += diffExt;
                }
            };

            if (A.empty()){
                //std::cout<<"return A empty\n";
                return .0;
            }

            const auto labelA = buffer.vertexLabels[A[0]];
            const auto labelB = (!B.empty()) ? buffer.vertexLabels[B[0]] : buffer.maxNotUsedLabel;

            computeDifferences(A, labelA, labelB);
            computeDifferences(B, labelB, labelA);
            gainFromMerging /= 2.0;

            // compute border
            border_.clear();
            for (const auto varA : A)
                if (buffer.referencedBy[varA] > 0)
                    border_.push_back(varA);
            for (const auto varB : B)
                if (buffer.referencedBy[varB] > 0)
                    border_.push_back(varB);



            moves_.clear();
            double cumulativeDiff = .0;
            std::pair<double, std::size_t> maxMove { std::numeric_limits<double>::lowest(), 0 };

            for (auto k = 0; k < settings_.numberOfIterations; ++k){
                //std::cout<<"k "<<k<<"\n";


                //for(const auto node: graph_.nodes()){
                //    //std::cout<<"buffer.differences[ "<<node<<" ] "<<buffer.differences[node]<<"\n";
                //}
                //for(const auto bvar: border_){
                //    //std::cout<<"border var "<<bvar<<"\n";
                //}

                Move m;
                if(B.empty() && k == 0){
                    for(const auto varA : A){
                        //std::cout<<"k0 or bEmpty\n";
                        m.updateDifference(varA, buffer.differences[varA]);
                    }
                }
                else{
                    uint64_t borderSize =  border_.size();
                    for (std::size_t i = 0; i < borderSize; ){

                        if (buffer.referencedBy[border_[i]] == 0)
                            std::swap(border_[i], border_[--borderSize]);
                        else{
                            //std::cout<<"else ..\n";
                            //std::cout<<"buffer var i .."<<border_[i]<<"\n";
                            //std::cout<<"buffer.differences there "<<buffer.differences[border_[i]]<<"\n";
                            m.updateDifference(border_[i], buffer.differences[border_[i]]);
                            ++i;
                        }
                    }
                    border_.erase(border_.begin() + borderSize, border_.end());
                }

                if(m.v == -1)
                    break;

                const auto oldLabel = buffer.vertexLabels[m.v];
                m.newLabel = (oldLabel == labelA ) ? labelB : labelA;

                // update differences and references
                for(const auto adj : liftedGraph_.adjacency(m.v)){

                    const auto adjNode = adj.node();
                    const auto adjEdge = adj.edge();

                    if(buffer.isMoved[adjNode])
                        continue;

                    const auto label = buffer.vertexLabels[adjNode];
                    if (label == m.newLabel){
                        //std::cout<<"e "<<adjEdge<<" w "<<weights_[adjEdge]<<"\n";
                        buffer.differences[adjNode] -= 2.0*weights_[adjEdge];
                    }
                    else if (label == oldLabel){
                        //std::cout<<"e "<<adjEdge<<" w "<<weights_[adjEdge]<<"\n";
                        buffer.differences[adjNode] += 2.0*weights_[adjEdge];
                    }
                }

                for(const auto adj : graph_.adjacency(m.v)){

                    const auto adjNode = adj.node();
                    const auto adjEdge = adj.edge();

                    if(buffer.isMoved[adjNode])
                        continue;

                    const auto label = buffer.vertexLabels[adjNode];

                    if (label == m.newLabel)
                        --buffer.referencedBy[adjNode];
                    
                    else if (label == oldLabel){
                        ++buffer.referencedBy[adjNode];
                        if (buffer.referencedBy[adjNode] == 1)
                            border_.push_back(adjNode);
                    }
                }


                buffer.vertexLabels[m.v] = m.newLabel;
                buffer.referencedBy[m.v] = 0;
                buffer.differences[m.v] = std::numeric_limits<double>::lowest();
                buffer.isMoved[m.v] = 1;
                moves_.push_back(m);

                cumulativeDiff += m.difference;
                if (cumulativeDiff > maxMove.first){
                    //std::cout<<"set max move with diff "<<cumulativeDiff<<"\n";
                    maxMove = std::make_pair(cumulativeDiff, moves_.size());
                }

            }



            if (gainFromMerging > maxMove.first && gainFromMerging > settings_.epsilon)
            {
                A.insert(A.end(), B.begin(), B.end());

                for (auto a : A)
                    buffer.vertexLabels[a] = labelA;

                for (auto b : B)
                    buffer.vertexLabels[b] = labelA;

                B.clear();

                //std::cout<<"return gainFromMerging\n";
                return gainFromMerging;
            }
            else if (maxMove.first > settings_.epsilon)
            {
                // revert some changes
                for (std::size_t i = maxMove.second; i < moves_.size(); ++i)
                {
                    buffer.isMoved[moves_[i].v] = 0;

                    if (moves_[i].newLabel == labelB)
                        buffer.vertexLabels[moves_[i].v] = labelA;
                    else
                        buffer.vertexLabels[moves_[i].v] = labelB;
                }

                // make sure that this is unique label
                if (B.empty())
                    ++buffer.maxNotUsedLabel;

                A.erase(std::partition(A.begin(), A.end(), [&](uint64_t a) { return !buffer.isMoved[a]; }), A.end());
                B.erase(std::partition(B.begin(), B.end(), [&](uint64_t b) { return !buffer.isMoved[b]; }), B.end());

                for (std::size_t i = 0; i < maxMove.second; ++i)
                    // move vertex to the other set
                    if (moves_[i].newLabel == labelB)
                        B.push_back(moves_[i].v);
                    else
                        A.push_back(moves_[i].v);

                //std::cout<<"return maxMove first\n";
                return maxMove.first;
            }
            else{
                for (std::size_t i = 0; i < moves_.size(); ++i)
                    if (moves_[i].newLabel == labelB)
                        buffer.vertexLabels[moves_[i].v] = labelA;
                    else
                        buffer.vertexLabels[moves_[i].v] = labelB;
            }

            //std::cout<<"return 0 at end\n";
            return .0;

        }




    private:
        const ObjectiveType & objective_;
        Settings settings_;
        const GraphType & graph_;
        const LiftedGraphType & liftedGraph_;
        const WeightsMap & weights_;
        std::vector<uint64_t> border_;
        std::vector<Move> moves_;
    };



} // end namespace detail_kernighang_lin






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
        typedef ComponentsBfs<GraphType> ComponentsType;
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
                return nodeLabels_[uv.first] == nodeLabels_[uv.second];
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
        LiftedMulticutKernighanLin(const ObjectiveType & objective, const Settings & settings = Settings());
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const ObjectiveType & objective() const;



        virtual const NodeLabels & currentBestNodeLabels( );
        virtual std::string name()const;
        virtual double currentBestEnergy();

    private:




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
        edges_()
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

        // components from current best node labels
        //std::cout<<"build cc\n";
        components_.build(GraphSubgraphWithCutFromNodeLabels(graph_, *currentBest_));
        //std::cout<<"done\n";

        for(const auto edge : graph_.edges()){
            const auto uv = graph_.uv(edge);
            const auto el = nodeLabels[uv.first] != nodeLabels[uv.second];
            const auto elc = components_[uv.first] != components_[uv.second];

            NIFTY_CHECK_OP(el,==,elc,"the input multicut labeling is invalid.");
        }






        // compute energy
        currentBestEnergy_ = objective_.evalNodeLabels(components_);



        // get maximum component index
        auto numberOfComponents = components_.maxLabel() + 1;


        // build the explicit partitions
        // and remember the last good/valid vertex labels\
    
        partitions_.clear();
        partitions_.resize(numberOfComponents);

        graph_.forEachNode([&](const uint64_t node){
            const auto ccLabel = components_[node];
            nodeLabels[node] = ccLabel;
            partitions_[ccLabel].push_back(node);
            twoCutBuffers_.vertexLabels[node] = ccLabel;
            lastGoodVertexLabels_[node] =  ccLabel;
        });
        twoCutBuffers_.maxNotUsedLabel = partitions_.size();

        for(const auto node : graph_.nodes()){
            visited_[node] = 0;
        }
        changed_.resize(numberOfComponents,1);
        std::fill(changed_.begin(), changed_.end(),1);

        // interatively update bipartition in order to minimize the total cost of the multicut
        for (size_t k = 0; k < settings_.numberOfOuterIterations; ++k){

            //std::cout<<"outer iteration "<<k<<"\n";

            auto energyDecrease = 0.0;

            // build components rag graph (just via edges)
            edges_.clear();
            edges_.resize(numberOfComponents);
            graph_.forEachEdge([&](const uint64_t edge){
                const auto uv = graph_.uv(edge);
                
                const auto lU = twoCutBuffers_.vertexLabels[uv.first];
                const auto lV = twoCutBuffers_.vertexLabels[uv.second];

                if(lU != lV){
                    edges_[std::min(lU,lV)].insert(std::max(lU,lV));
                }
                
            });

            for(uint64_t piU=0; piU<numberOfComponents; ++piU){
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

            auto ee = energyDecrease;

            // remove partitions that became empty after the previous step
            auto partionF =  [](const std::vector<size_t>& s) { return !s.empty(); };
            auto newEnd = std::partition(partitions_.begin(), partitions_.end(), partionF);
            partitions_.resize(newEnd - partitions_.begin());


            NIFTY_CHECK_OP(partitions_.size(), > ,0, "");

            // try to intoduce new partitions 
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


            if (energyDecrease == .0){
                //std::cout<<"energyDecrease == 0 \n";
                break;
            }
        
            std::stack<std::size_t> S;
            for (const auto node : graph_.nodes()){
                visited_[node] = 0;
            }

            // do connected component labeling on the original graph
            numberOfComponents = 0;

            for (const auto i : graph_.nodes()){
                if (!visited_[i]){

                    S.push(i);
                    visited_[i] = 1;

                    const auto label = twoCutBuffers_.vertexLabels[i];

                    twoCutBuffers_.referencedBy[i] = numberOfComponents;

                    while (!S.empty()){

                        auto v = S.top();
                        S.pop();

                        for(const auto adj : graph_.adjacency(v)){

                            if (twoCutBuffers_.vertexLabels[adj.node()] == label && !visited_[adj.node()]){
                                S.push(adj.node());
                                visited_[adj.node()] = 1;
                                twoCutBuffers_.referencedBy[adj.node()] = numberOfComponents;
                            }
                        }
                    }

                    ++numberOfComponents;
                }
            }
           
            // compute new true energy
            //double newEnergyValue = .0;
            double newEnergyValue = objective_.evalNodeLabels(twoCutBuffers_.referencedBy);
            /*
            for (const auto edge : liftedGraph_.edges()){
                const auto uv = liftedGraph_.uv(edge);
                if (twoCutBuffers_.referencedBy[uv.first] != twoCutBuffers_.referencedBy[uv.second])
                    newEnergyValue += objective_.weights()[edge];
            }
            */

            //std::cout<<"new E "<<newEnergyValue<<" oldE "<<currentBestEnergy_<<" eps"<<settings_.epsilon<<"\n";



            // if the new true energy is higher, than the current one, revert the changes and terminate
            if (newEnergyValue >= currentBestEnergy_ - settings_.epsilon)
            {
                //for (const auto edge : liftedGraph_.edges()){
                //    const auto uv = liftedGraph_.uv(edge);
                //    nodeLabels[edge] = lastGoodVertexLabels_[uv.first] == lastGoodVertexLabels_[uv.second] ? 0 : 1;
                //}
                for(const auto node : graph_.nodes()){
                    nodeLabels[node] = lastGoodVertexLabels_[node];
                }
                break;
            }

            // otherwise, form new partitions
            partitions_.clear();
            partitions_.resize(numberOfComponents);

            for (const auto i : graph_.nodes()){
                twoCutBuffers_.vertexLabels[i] = twoCutBuffers_.referencedBy[i];
                partitions_[twoCutBuffers_.vertexLabels[i]].push_back(i);
            }



            twoCutBuffers_.maxNotUsedLabel = numberOfComponents;


            for (const auto i : graph_.nodes()){
                nodeLabels[i] = twoCutBuffers_.vertexLabels[i];
            }
            bool didntChange = true;
            for (const auto i : liftedGraph_.edges()){
                const auto uv = liftedGraph_.uv(i);
                const auto v0 = uv.first;
                const auto v1 = uv.second;
                const auto edgeLabel  = (twoCutBuffers_.vertexLabels[v0] == twoCutBuffers_.vertexLabels[v1]) ? 0 : 1;

                if (static_cast<bool>(edgeLabel) != (lastGoodVertexLabels_[v0] != lastGoodVertexLabels_[v1]))
                    didntChange = false;
            }
            if (didntChange)
                break;





            // check if the shape of some partitions didn't change
                //changed.clear();
            changed_.clear();
            changed_.resize(numberOfComponents);
            std::fill(changed_.begin(), changed_.end(), 0);

            for (const auto node : graph_.nodes()){
                visited_[node] = 0;
            }


            for (const auto i : graph_.nodes())
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

            for(const auto  node : graph_.nodes())
                lastGoodVertexLabels_[node] = twoCutBuffers_.vertexLabels[node];

            // call visitor here

            //if (!visitor(lastGoodVertexLabels_) || visitor.time_limit_exceeded())
            //    break;

            //if (1)
            //    std::cout << std::setw(4) << k+1 << std::setw(16) << currentBestEnergy_ - newEnergyValue << std::setw(15) << energyDecrease << std::setw(15) << ee << std::setw(15) << (energyDecrease - ee) << std::setw(14) << partitions_.size() << std::endl;


            currentBestEnergy_ = newEnergyValue;




            // call visitor
            if(!visitorProxy.visit(this))
                break;










        }
        visitorProxy.end(this);     
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

#endif  // NIFTY_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_KERNIGHAN_LIN_HXX
