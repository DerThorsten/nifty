#pragma once
#ifndef NIFTY_GRAPH_BIDIRECTIONAL_BREADTH_FIRST_SEARCH_HXX
#define NIFTY_GRAPH_BIDIRECTIONAL_BREADTH_FIRST_SEARCH_HXX

#include <cstddef>
#include <limits> // std::numeric_limits
#include <deque>
#include <queue>
#include <vector>
#include <algorithm> // std::reverse
                     
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/detail/search_impl.hxx"
#include "nifty/queue/changeable_priority_queue.hxx"

namespace nifty{
namespace graph{

    namespace detail_bfs{

        template<class T>
        inline void
        singleSourceSingleTargetHelper(
            const std::vector<std::ptrdiff_t>& parents,
            const T vPositive,
            const T vNegative,
            std::deque<T>& path
        ) {
            assert(vPositive >= 0);
            assert(vNegative >= 0);
            T t = vPositive;
            for(;;) {
                path.push_front(t);
                if(parents[t] - 1 == t) {
                    break;
                }
                else {
                    t = parents[t] - 1;
                }
            }
            t = vNegative;
            for(;;) {
                path.push_back(t);
                if(-parents[t] - 1 == t) {
                    break;
                }
                else {
                    t = -parents[t] - 1;
                }
            }
        }

    };  // end namespace detail bfs

    template<class GRAPH>
    class BidirectionalBreadthFirstSearch{
    private:
        enum RetFlag{
            ReturnTrue,
            ReturnFalse,
            ContinueSeach
        };

    public:
        typedef GRAPH Graph;
        typedef typename Graph:: template NodeMap<int64_t> Parents;

        BidirectionalBreadthFirstSearch(const Graph & graph)
        :   graph_(graph),
            parents_(graph){

        }


        bool runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target
        ){
            DefaultSubgraphMask<Graph> subgraphMask;
            return this->runSingleSourceSingleTarget(source,target,subgraphMask);
        }

    


        template<class SUBGRAPH_MASK>
        bool runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target,
            const SUBGRAPH_MASK & mask
        ){
            path_.clear();
            if(!mask.useNode(source) || !mask.useNode(target))
                return false;
            if(source == target){
                path_.push_back(source);
                return true;
            }
            std::fill(parents_.begin(), parents_.end(), 0);
            parents_[source] = source + 1;
            parents_[target] =  -static_cast<int64_t>(target) - 1;

            // clear queues
            for(auto q=0; q<2; ++q)
                while(!queues_[q].empty())
                    queues_[q].pop();

            queues_[0].push(source);
            queues_[1].push(target);


            for(auto q = 0; true; q = 1 - q) { // infinite loop, alternating queues
                const auto numberOfNodesAtFront = queues_[q].size();
                for(auto n = 0; n < numberOfNodesAtFront; ++n) {
                    auto v = queues_[q].front();
                    queues_[q].pop();


                    auto fHelper = [&](const uint64_t otherNode, const uint64_t edge){
                        if(!mask.useEdge(edge) || !mask.useNode(otherNode))
                            return ContinueSeach;

                        const auto p = parents_[otherNode];
                        if(p < 0 && q == 0){
                            detail_bfs::singleSourceSingleTargetHelper(parents_, v, otherNode, path_);
                            return ReturnTrue;
                        }
                        else if(p > 0 && q == 1){
                            detail_bfs::singleSourceSingleTargetHelper(parents_, otherNode, v, path_);
                            return ReturnTrue;
                        }
                        else if(p == 0){
                            if(q == 0)
                                parents_[otherNode] = v + 1;
                            else
                                parents_[otherNode] = -static_cast<int64_t>(v) - 1;
                            queues_[q].push(otherNode);
                        }

                    };

                    if(q == 0){
                        for(auto adj : graph_.adjacencyOut(v)){
                            auto retFlag = fHelper(adj.node(), adj.edge());
                            if(retFlag == ContinueSeach)
                                continue;
                            else if(retFlag == ReturnTrue)
                                return true;
                        }
                    }
                    else{
                        for(auto adj : graph_.adjacencyIn(v)){
                            auto retFlag = fHelper(adj.node(), adj.edge());
                            if(retFlag == ContinueSeach)
                                continue;
                            else if(retFlag == ReturnTrue)
                                return true;
                        }          
                    }
                }
                if(queues_[0].empty() && queues_[1].empty()) {
                    return false;
                }
            }

        }
        const std::deque<uint64_t> & path() const {
            return path_;
        }
    private:




        const Graph & graph_;
        Parents parents_;
        std::array< std::queue<uint64_t> , 2> queues_;
        std::deque<uint64_t> path_;
    };
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_BIDIRECTIONAL_BREADTH_FIRST_SEARCH_HXX
