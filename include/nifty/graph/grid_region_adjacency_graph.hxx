#pragma once
#ifndef NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX
#define NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX

#include <algorithm>

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
        void assignLabels(const nifty::marray::View<T> & labels){

            //std::cout<<"find max \n";

            auto nLabels = *std::max_element(labels.begin(),labels.end())+1;
            this->assign(nLabels);

            const std::vector<uint64_t> shape(labels.shapeBegin(),labels.shapeEnd());
            const auto nZ = shape[2];

            typedef std::pair< int64_t, int64_t> Edge;
            typedef std::set<Edge> EdgeSet;
            typedef std::vector<EdgeSet> EdgeSetVec;

            EdgeSetVec betweenSliceEdgesVec(nZ-1);
            EdgeSetVec inSliceEdgesVec(nZ);

            nifty::parallel::ThreadPool  threadpool(-1);
            
            auto insertEdgeIfDifferent = [](EdgeSet & edgeSet, T l, T ol){
                if(l != ol){
                    const auto e = Edge(std::min(l, ol), std::max(l, ol));
                    edgeSet.insert(e);
                }
            };

            //std::cout<<"in slices edges \n";
            // find in slice edges
            nifty::parallel::parallel_foreach(threadpool,nZ,[&]
            (int threadIndex,int z){
                const auto labels2d = labels.boundView(2,z);
                auto & inSliceEdges = inSliceEdgesVec[z];

                for(auto y=0; y<shape[1]; ++y)
                for(auto x=0; x<shape[0]; ++x){
                    const auto l = labels2d(x, y);
                    if(x + 1 < shape[0]){
                        insertEdgeIfDifferent(inSliceEdges, l, labels2d(x + 1, y));
                    }
                    if(y + 1 < shape[1]){
                        insertEdgeIfDifferent(inSliceEdges, l, labels2d(x, y + 1));
                    }
                }
            });
            //std::cout<<"between slice edges \n";
            // find between slice edges
            nifty::parallel::parallel_foreach(threadpool,nZ-1,[&]
            (int threadIndex,int z){
                const auto labelsA = labels.boundView(2,z);
                const auto labelsB = labels.boundView(2,z+1);
                auto & betweenSliceEdges = betweenSliceEdgesVec[z];
                for(auto y=0; y<shape[1]; ++y)
                for(auto x=0; x<shape[0]; ++x){
                    insertEdgeIfDifferent(betweenSliceEdges, labelsA(x, y), labelsB(x, y));
                }
            });

            //std::cout<<"add edges in slices\n";
            // add edges in serial
            for(const auto & inSliceEdges  :inSliceEdgesVec){
                for(const auto & edge : inSliceEdges){
                    this->insertEdge(edge.first,edge.second);
                }
            }

            //std::cout<<"add edges between slices\n";
            // add edges in serial
            for(const auto & betweenSliceEdges  :betweenSliceEdgesVec){
                for(const auto & edge : betweenSliceEdges){
                    this->insertEdge(edge.first,edge.second);
                }
            }

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
