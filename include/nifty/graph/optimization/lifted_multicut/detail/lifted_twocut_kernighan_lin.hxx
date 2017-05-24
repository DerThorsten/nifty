// reimplementation of kerninhanlin in 
// https://github.com/bjoern-andres/graph

#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_DETAIL_LIFTED_TWOCUT_KERNIGHAN_LIN_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_DETAIL_LIFTED_TWOCUT_KERNIGHAN_LIN_HXX


#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <stack>

#include "nifty/tools/changable_priority_queue.hxx"

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/components.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{

// \cond SUPPRESS_DOXYGEN
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
// \endcond

} // lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_DETAIL_LIFTED_TWOCUT_KERNIGHAN_LIN_HXX
