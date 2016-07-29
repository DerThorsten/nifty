#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX


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


#include "nifty/graph/rag/grid_rag_labels_chunked.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{


template<class LABELS_PROXY>
class GridRagSliced;


namespace detail_rag{

template<class LABEL_TYPE>
struct ComputeRag< GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> > {
    
    static void computeRag(
        GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> & rag,
        const typename GridRagSliced<ChunkedLabels<3, LABEL_TYPE>>::Settings & settings ){
        
        typedef GridRagSliced<ChunkedLabels<3, LABEL_TYPE>> Graph;
        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        const auto & labelsProxy = rag.labelsProxy();
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
        }
    }


};


} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_CHUNKED_HXX */
