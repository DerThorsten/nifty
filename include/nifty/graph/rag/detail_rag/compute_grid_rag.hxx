#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX


#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>
#include <unordered_set>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif


#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class ExplicitLabels;

template<size_t DIM, class LABELS_PROXY>
class GridRag;

template<class LABELS_PROXY>
class GridRagSliced;




namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){
        typedef GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> >  Graph;
        typedef std::array<int64_t, DIM> Coord;
        typedef typename Graph::NodeAdjacency NodeAdjacency;
        typedef typename Graph::EdgeStorage EdgeStorage;

        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto & labels = labelsProxy.labels(); 
        const auto & shape = labelsProxy.shape();

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        rag.assign(numberOfLabels);

        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        if(pOpts.getActualNumThreads()<=1){
            nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
                const auto lU = labels(coord);
                for(size_t axis=0; axis<DIM; ++axis){
                    auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2);
                        if(lU != lV){
                            rag.insertEdge(lU,lV);
                        }
                    }
                }
            });
        }
        else{
            nifty::parallel::ThreadPool threadpool(pOpts);
            struct PerThread{
                std::vector< __setimpl<uint64_t> > adjacency;
            };

            std::vector<PerThread> perThreadDataVec(pOpts.getActualNumThreads());
            for(size_t i=0; i<perThreadDataVec.size(); ++i)
                perThreadDataVec[i].adjacency.resize(numberOfLabels);

            // collect the node-adjacency sets in parallel which needs to be merged later 
            nifty::tools::parallelForEachCoordinate(threadpool, shape,[&](const int tid, const Coord & coord){
                auto & adjacency = perThreadDataVec[tid].adjacency;
                const auto lU = labels(coord);
                for(size_t axis=0; axis<DIM; ++axis){
                    auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2);
                        if(lU != lV){
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
            
            // merge the node adjacency sets for each node
            nifty::parallel::parallel_foreach(threadpool, numberOfLabels, [&](int tid, int label){
                auto & set0 = perThreadDataVec[0].adjacency[label];
                for(size_t i=1; i<perThreadDataVec.size(); ++i){
                    const auto & setI = perThreadDataVec[i].adjacency[label];
                    set0.insert(setI.begin(), setI.end());
                }
                for(auto otherNode : set0)
                     rag.nodes_[label].insert(NodeAdjacency(otherNode));
            });

            // insert the edge index for each edge
            uint64_t edgeIndex = 0;
            auto & edges = rag.edges_;
            for(uint64_t u = 0; u< numberOfLabels; ++u){
                for(auto & vAdj :  rag.nodes_[u]){
                    const auto v = vAdj.node();
                    if(u < v){
                        edges.push_back(EdgeStorage(u, v));
                        vAdj.changeEdgeIndex(edgeIndex);
                        auto fres =  rag.nodes_[v].find(NodeAdjacency(u));
                        fres->changeEdgeIndex(edgeIndex);
                        ++edgeIndex;
                    }
                }
            }
        }
    }
};

template<class LABEL_TYPE>
struct ComputeRag< GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> > {
    
    static void computeRag(
        GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> & rag,
        const typename GridRagSliced<ChunkedLabels<3, LABEL_TYPE>>::Settings & settings ){
        
        typedef GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> Graph;
        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto & labels = labelsProxy.labels(); 
        
        rag.assign(numberOfLabels);

        size_t z_max = labels.shape(0);
        size_t y_max = labels.shape(1);
        size_t x_max = labels.shape(2);

        vigra::Shape3 slice_shape(1, y_max, x_max);

        vigra::MultiArray<3,LABEL_TYPE> this_slice(slice_shape);
        vigra::MultiArray<3,LABEL_TYPE> next_slice(slice_shape);

        // TODO parallel versions of the code

        for(size_t z = 0; z < z_max; z++) {
            
            // checkout this slice
            vigra::Shape3 slice_begin(z, 0, 0);
            labels.checkoutSubarray(slice_begin, this_slice);

            if(z < z_max - 1) {
                // checkout next slice
                vigra::Shape3 next_begin(z+1, 0, 0);
                labels.checkoutSubarray(next_begin, next_slice);
            }

            // single core
            if(pOpts.getActualNumThreads()<=1){
            
                for(size_t y = 0; y < y_max; y++) {
                    for(size_t x = 0; x < x_max; x++) {
                        
                        const auto lu = this_slice(0,y,x);
                        
                        if(x < x_max-1) {
                            const auto lv = this_slice(0,y,x+1);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }
                        
                        if(y < y_max-1) {
                            const auto lv = this_slice(0,y+1,x);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }

                        if(z < z_max-1) {
                            const auto lv = next_slice(0,y,x);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }
                    }
                }
            }

            // FIXME this is slower AND does not produce the correct result
            // multi core, locked
            // Thorstens idea: first iterate over all the slices in parallel, extracting only in slice edges, then extract inter slice edges, this has the advantage that we can also extract a dividing index between xy and z edges 
            //else if(!settings.lockFreeAlg){
            //    std::mutex mutexArray[5000];
            //    std::mutex edgeMutex;
            //    nifty::parallel::ThreadPool threadpool(pOpts);
            //    nifty::parallel::parallel_foreach(threadpool, y_max,
            //    [&](int tid, int y){

            //        for(size_t x=0; x<x_max; ++x){
            //            
            //            const auto lu = this_slice(0,y,x);
            //            
            //            if(x+1<x_max){
            //                const auto lv = this_slice(0,y,x+1);
            //                if(lu != lv){
            //                    rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
            //                }
            //            }
            //            if(y+1<y_max){
            //                const auto lv = this_slice(0,y+1,x);
            //                if(lu != lv){
            //                    rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
            //                }
            //            }
            //            if(z+1<z_max){
            //                const auto lv = next_slice(0, y, x);
            //                if(lu != lv){
            //                    rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
            //                }
            //            }
            //        }

            //    });
            //}
        }
    }


};



} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX */
