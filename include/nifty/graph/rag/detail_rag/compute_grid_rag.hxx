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


template<class LABEL_TYPE>
struct ComputeRag< GridRag<2,  ExplicitLabels<2, LABEL_TYPE> > > {
    template<class S>
    static void computeRag(
        GridRag<2,  ExplicitLabels<2, LABEL_TYPE> > & rag,
        const S & settings
    ){
        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto labels = labelsProxy.labels(); 
        
        // assign the number of nodes to the graph
        rag.assign(numberOfLabels);

        for(size_t x=0; x<labels.shape(0); ++x)
        for(size_t y=0; y<labels.shape(1); ++y){

            const auto lu = labels(x, y);
            if(x+1<labels.shape(0)){
                const auto lv = labels(x+1, y);
                if(lu != lv){
                    rag.insertEdge(lu,lv);
                }
            }
            if(y+1<labels.shape(1)){
                const auto lv = labels(x, y+1);
                if(lu != lv){
                    rag.insertEdge(lu,lv);
                }
            }
        }
    }
};

template<class LABEL_TYPE>
struct ComputeRag< GridRag<3,  ExplicitLabels<3, LABEL_TYPE> > > { 
    static void computeRag(
        GridRag<3,  ExplicitLabels<3, LABEL_TYPE> > & rag,
        const typename GridRag<3,  ExplicitLabels<3, LABEL_TYPE> >::Settings & settings
    ){
        typedef GridRag<3,  ExplicitLabels<3, LABEL_TYPE> >  Graph;
        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto labels = labelsProxy.labels(); 
        
        
        rag.assign(numberOfLabels);
        if(pOpts.getActualNumThreads()<=1){
                 
            for(size_t x=0; x<labels.shape(0); ++x)
            for(size_t y=0; y<labels.shape(1); ++y)
            for(size_t z=0; z<labels.shape(2); ++z){
                const auto lu = labels(x, y, z);
                if(x+1<labels.shape(0)){
                    const auto lv = labels(x+1, y, z);
                    if(lu != lv){
                        rag.insertEdge(lu,lv);
                    }
                }
                if(y+1<labels.shape(1)){
                    const auto lv = labels(x, y+1, z);
                    if(lu != lv){
                        rag.insertEdge(lu,lv);
                    }
                }
                if(z+1<labels.shape(2)){
                    const auto lv = labels(x, y, z+1);
                    if(lu != lv){
                        rag.insertEdge(lu,lv);
                    }
                }
            }
        }
        else if(!settings.lockFreeAlg){
            std::mutex mutexArray[5000];
            std::mutex edgeMutex;
            nifty::parallel::ThreadPool threadpool(pOpts);
            nifty::parallel::parallel_foreach(threadpool, labels.shape(0),
            [&](int tid, int x){

                for(size_t y=0; y<labels.shape(1); ++y)
                for(size_t z=0; z<labels.shape(2); ++z){
                    const auto lu = labels(x, y, z);
                    if(x+1<labels.shape(0)){
                        const auto lv = labels(x+1, y, z);
                        if(lu != lv){
                            //++e;
                            rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
                        }
                    }
                    if(y+1<labels.shape(1)){
                        const auto lv = labels(x, y+1, z);
                        if(lu != lv){
                            //++e;
                            rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
                        }
                    }
                    if(z+1<labels.shape(2)){
                        const auto lv = labels(x, y, z+1);
                        if(lu != lv){
                            //++e;
                            rag.inserEdgeWithMutex(lu,lv, edgeMutex, mutexArray, 5000);
                        }
                    }
                }

            });
        }
        else{
            struct PerThread{
                std::vector< __setimpl<uint64_t> > adjacency;
            };

            std::vector<PerThread> perThreadDataVec(pOpts.getActualNumThreads());
            for(size_t i=0; i<perThreadDataVec.size();++i){
                perThreadDataVec[i].adjacency.resize(numberOfLabels);
            }
            nifty::parallel::ThreadPool threadpool(pOpts);

            nifty::parallel::parallel_foreach(threadpool, labels.shape(0),
            [&](int tid, int x){

                auto & perThreadData = perThreadDataVec[tid];
                auto & adjacency = perThreadData.adjacency;

                auto fEdgelet = [&](const LABEL_TYPE la, const LABEL_TYPE lb){
                    if(la!=lb){
                        adjacency[la].insert(lb);
                        adjacency[lb].insert(la);
                    }
                };

                for(size_t y=0; y<labels.shape(1); ++y)
                for(size_t z=0; z<labels.shape(2); ++z){
                    const auto lu = labels(x, y, z);
                    if(x+1<labels.shape(0)){
                        fEdgelet(lu,labels(x+1, y, z));
                    }
                    if(y+1<labels.shape(1)){
                        fEdgelet(lu,labels(x, y+1, z));
                    }
                    if(z+1<labels.shape(2)){
                        fEdgelet(lu,labels(x, y, z+1));
                    }
                }
            });
            
            typedef typename Graph::NodeAdjacency NodeAdjacency;
            typedef typename Graph::EdgeStorage EdgeStorage;
            auto & ragNodesAdj  = rag.nodes_;
            
            nifty::parallel::parallel_foreach(threadpool, numberOfLabels,
            [&](int tid, int label){

                auto & set = ragNodesAdj[label];
                auto & set0 = perThreadDataVec[0].adjacency[label];
                for(size_t i=1; i<perThreadDataVec.size(); ++i){
                    const auto & setI = perThreadDataVec[i].adjacency[label];
                    set0.insert(setI.begin(), setI.end());
                }
                for(auto otherNode : set0){
                    set.insert(NodeAdjacency(otherNode));
                }
            });

            uint64_t edgeIndex = 0;
            auto & edges = rag.edges_;
            for(uint64_t u = 0; u< numberOfLabels; ++u){
                auto & adjSetU = ragNodesAdj[u];
                for(auto & vAdj : adjSetU){
                    const auto v = vAdj.node();
                    if(u <  v){
                        edges.push_back(EdgeStorage(u, v));
                        vAdj.changeEdgeIndex(edgeIndex);
                        auto & adjSetV = ragNodesAdj[v];
                        auto fres = adjSetV.find(NodeAdjacency(u));
                        fres->changeEdgeIndex(edgeIndex);
                        ++edgeIndex;
                    }
                    else{
                        // do nothing
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
        const auto labels = labelsProxy.labels(); 

        std::cout << "Computing Chunked RAG" << std::endl;
        std::cout << numberOfLabels << std::endl;
        std::cout << *std::max_element(labels.cbegin(), labels.cend())+1 << std::endl;
        
        rag.assign(numberOfLabels);

        // Loop over the chunks of the label volume and extract the edges
        // FIXME We need an overlap of 1 in x / y blocks to make sure that we hit all edges (maybe should jsut cache these before going to the next chunk)
        // TODO parallelize (over z axis or inner ?)
        // TODO benchmark and optimize access pattern
        
        // access pattern:
        // We assumes 2d like chunks, e.g. (512,512,1) or (1024,1024,1)
        // we then loop over z and check out the chunks in z and z + 1, looping over them to find all labels

        size_t z_max = labels.shape(0);
        size_t y_max = labels.shape(1);
        size_t x_max = labels.shape(2);

        //std::cout << x_max << std::endl;
        //std::cout << y_max << std::endl;
        //std::cout << z_max << std::endl;

        //std::cout << labels.chunkShape(0) << std::endl;
        //std::cout << labels.chunkShape(1) << std::endl;
        //std::cout << labels.chunkShape(2) << std::endl;

        for(size_t z = 0; z < z_max - 1; z++) {
            
            // chunks in this slice
            vigra::Shape3 roi_s(z,0,0), roi_e(z+1,y_max,x_max);
            auto chunk = labels.chunk_begin(roi_s,roi_e), end = labels.chunk_end(roi_s,roi_e);
            
            // chunks in the next slice
            vigra::Shape3 roi_s_up(z+1,0,0), roi_e_up(z+2,y_max,x_max);
            auto chunk_up = labels.chunk_begin(roi_s_up,roi_e_up);

            for(; chunk != end; ++chunk, ++chunk_up) {
                
                vigra::MultiArrayView<3, LABEL_TYPE> chunk_view = *chunk;
                vigra::MultiArrayView<3, LABEL_TYPE> chunk_view_up = *chunk_up;
                
                for(size_t y=0; y<chunk_view.shape(1); ++y) {
                    for(size_t x=0; x<chunk_view.shape(2); ++x) {
                    
                        const auto lu = chunk_view(0,y,x);
                        if(x+1<labels.shape(0)){
                            const auto lv = chunk_view(0, y, x+1);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }
                        if(y+1<chunk_view.shape(1)){
                            const auto lv = chunk_view(0, y+1, x);
                            if(lu != lv)
                                rag.insertEdge(lu,lv);
                        }
                        const auto lv = chunk_view_up(0, y, x);
                        if(lu != lv)
                            rag.insertEdge(lu,lv);
                    }
                }
            }
        }
        // chunks in the last slice
        vigra::Shape3 roi_s(z_max-1,0,0), roi_e(z_max,y_max,x_max);
        auto chunk = labels.chunk_begin(roi_s,roi_e), end = labels.chunk_end(roi_s,roi_e);
            
        for(; chunk != end; ++chunk) {
            
            vigra::MultiArrayView<3, LABEL_TYPE> chunk_view = *chunk;
            
            for(size_t y=0; y<chunk_view.shape(1); ++y) {
                for(size_t x=0; x<chunk_view.shape(2); ++x) {
                
                    const auto lu = chunk_view(0, y, x);
                    if(x+1<labels.shape(0)){
                        const auto lv = chunk_view(y, x+1);
                        if(lu != lv){
                            rag.insertEdge(lu,lv);
                        }
                    }
                    if(y+1<chunk_view.shape(1)){
                        const auto lv = chunk_view(y+1, x);
                        if(lu != lv){
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


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX */
