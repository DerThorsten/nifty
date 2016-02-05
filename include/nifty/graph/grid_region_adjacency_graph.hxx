#pragma once
#ifndef NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX
#define NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX

#include <algorithm>

#include <boost/iterator/transform_iterator.hpp>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/tools/const_iterator_range.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/features/accumulated_features.hxx"


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

        /**
         * @brief      accumulated statistics over edges
         * 
         *
         * @param[in]  labels  input labels which generated rag
         * @param[in]  data    pixel wise data 
         *
         * @tparam     LABELS  pixel wise label input type, must fulfill the slice-vol requirements
         * @tparam     DATA    pixel wise data input type, must fulfill the slice-vol requirements
         * 
         *   Compute statistics as :
         *      - Mean
         *      - Min,Max
         *      - Quantiles (0.1,0.25,0.5,0.75,0.9)
         *      - StdDev
         *      - Kurtosis
         *    Over all nodes and edges
         *    And derive features from that.
         *    the features will take min mean sum prod etc. between
         *    adjacent nodes to generate edge features from node features.
         *    In Total we have 11 features.
         *    
         * 
         */
        template<
            class LABELS,
            class DATA,
            class T
        >
        void accumulateEdgeFeatures(
            const LABELS & labels,
            const DATA   & data,
            nifty::marray::View<T> & features
        )const{
            const auto nZ  = shape_[2];
            nifty::parallel::ThreadPool  threadpool(-1);

            std::vector<nifty::features::DefaultAccumulatedStatistics<T> >  accs_(this->numberOfEdges());


            // do the accumulation
            nifty::parallel::parallel_foreach(threadpool,nZ,[&](int threadIndex,int z){

                auto labelsZ = labels.slice(z);
                auto dataZ = data.slice(z);

                // little lambda to make code cleaner
                auto accInSliceEdges = [&](const int x,const int y){
                    const auto l = labelsZ(x, y);
                    if(x + 1 < shape_[0]){
                        const auto ol = labelsZ(x+1, y);
                        if(l!=ol)
                            accs_[this->findEdge(l,ol)].acc(dataZ(x,y)).acc(dataZ(x+1,y));
                    }
                    if(y + 1 < shape_[1]){
                        const auto ol = labelsZ(x, y+1);
                        if(l!=ol)
                            accs_[this->findEdge(l,ol)].acc(dataZ(x,y)).acc(dataZ(x,y+1));
                    }
                };

                if(z+1<nZ){
                    auto labelsZ1 = labels.slice(z+1);
                    auto dataZ1 = data.slice(z+1);
                    for(auto y=0; y<shape_[1]; ++y)
                    for(auto x=0; x<shape_[0]; ++x){
                        const auto l = labelsZ(x, y);
                        // in slide edges
                        accInSliceEdges(x,y);
                        // between slide edges
                        accs_[this->findEdge(l, labelsZ1(x,y))].acc(dataZ(x,y)).acc(dataZ1(x,y));
                    }
                }
                else{
                    for(auto y=0; y<shape_[1]; ++y)
                    for(auto x=0; x<shape_[0]; ++x){
                        accInSliceEdges(x,y);
                    }
                }
            });

            // store the resulting features
            nifty::parallel::parallel_foreach(threadpool,this->numberOfEdges(),[&](int threadIndex,int edge){
                auto featureEdge = features.bindView(1,edge);
                accs_[edge].result(featureEdge.begin(),featureEdge.end());
            });

            // and we are done
        }


        
    private:

        nifty::array::StaticArray<size_t ,3> shape_;

        std::vector<std::pair<uint64_t,uint64_t> > inSliceNodeRanges_;
        std::vector<std::pair<uint64_t,uint64_t> > inSliceEdgeRanges_;
        std::vector<std::pair<uint64_t,uint64_t> > betweenSliceEdgeRanges_;
    };



    /*
     *  Rag 3d2d features
     *
     *
     */





} // namespace nifty::graph
} // namespace nifty
  // 
#endif  // NIFTY_GRID_REGION_ADJACENCY_GRAPH_HXX
