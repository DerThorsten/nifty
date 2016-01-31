#pragma once
#ifndef NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX
#define NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX

#include <boost/iterator/transform_iterator.hpp>

#include "nifty/marray/marray.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/tools/const_iterator_range.hxx"
#include "nifty/parallel/threadpool.hxx"


namespace nifty{
namespace graph{



    class Rag3d : public UndirectedGraph< >{

    };


    class Rag3d2d : public Rag3d{

        typedef typename Rag3d::EdgeIter EdgeIter;
        typedef typename Rag3d::NodeIter NodeIter;

        typedef tools::ConstIteratorRange<EdgeIter> EdgeRange;
        typedef tools::ConstIteratorRange<NodeIter> NodeRange;

    public:

        template<class T>
        void assign(const nifty::marray::View<T> & labels){
            const auto shape = labels.shape();
            const auto nZ = shape[2];

            typedef std::pair< int64_t, int64_t> Edge;
            typedef std::set<Edge> EdgeSet;
            typedef std::vector<EdgeSet> EdgeSetVec;

            EdgeSetVec betweenSliceEdgesVec(nZ-1);
            EdgeSetVec inSliceEdgesVec(nZ);

            nifty::parallel::ThreadPool  threadpool(-1);
            
            auto insertEdgeIfDifferent = [](EdgeSet & edgeSet, int64_t l, int64_t ol){
                if(l != ol){
                    const auto e = Edge(std::min(l, ol), std::max(l, ol));
                    edgeSet.insert(e);
                }
            };

            // find in slice edges
            nifty::parallel::parallel_foreach(threadpool,nZ,[&]
            (int threadIndex,int z){
                const size_t begin[3] = {0,0,size_t(z)};
                const size_t shape[3] = {size_t(shape[0]),size_t(shape[1]),1}; 
                const auto labels2d = labels.view(begin, shape).squeeze();
                auto & inSliceEdges = inSliceEdgesVec[z];

                for(auto y=0; y<shape[1]; ++y)
                for(auto x=0; x<shape[0]; ++x){
                    const auto l = labels(x, y);
                    if(x + 1 < shape[0]){
                        insertEdgeIfDifferent(l, labels(x + 1, y));
                    }
                    if(y + 1 < shape[1]){
                        insertEdgeIfDifferent(l, labels(x, y + 1));
                    }
                }
            });

            // find between slice edges
            nifty::parallel::parallel_foreach(threadpool,nZ-1,[&]
            (int threadIndex,int z){
                const size_t beginA[3] = {0, 0, size_t(z)};
                const size_t beginB[3] = {0, 0, size_t(z+1)};
                const size_t shape[3] = {size_t(shape[0]),size_t(shape[1]),1}; 
                const auto labelsA = labels.view(beginA, shape).squeeze();
                const auto labelsB = labels.view(beginA, shape).squeeze();
                auto & betweenSliceEdges = betweenSliceEdgesVec[z];
                for(auto y=0; y<shape[1]; ++y)
                for(auto x=0; x<shape[0]; ++x){
                    insertEdgeIfDifferent(labelsA(x, y), labelsB(x, y));
                }
            });

            // 

        }

        // extra members
        NodeRange inSliceNodes(const uint64_t sliceIndex)const{
            const auto beginIter = this->nodesBegin();
            const auto & r = inSliceNodeRanges_[sliceIndex];
            return NodeRange(beginIter+r.first, beginIter+r.second);
        }
        EdgeRange inSliceEdges(const uint64_t sliceIndex)const{
            const auto beginIter = this->edgesBegin();
            const auto & r = inSliceEdgeRanges_[sliceIndex];
            return EdgeRange(beginIter+r.first, beginIter+r.second);
        }
        EdgeRange betweenSliceEdges(const uint64_t sliceIndex)const{
            const auto beginIter = this->edgesBegin();
            const auto & r = betweenSliceEdgeRanges_[sliceIndex];
            return EdgeRange(beginIter+r.first, beginIter+r.second);
        }

    private:


        std::vector<std::pair<uint64_t,uint64_t> > inSliceNodeRanges_;
        std::vector<std::pair<uint64_t,uint64_t> > inSliceEdgeRanges_;
        std::vector<std::pair<uint64_t,uint64_t> > betweenSliceEdgeRanges_;
    };



} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX
