#pragma once

#include <functional>
#include <algorithm>
#include <map>
#include <cstddef>

#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_coordinate.hxx"


namespace nifty{
namespace graph{


template<std::size_t DIM, class LABEL_TYPE>
class ExplicitLabels;

template<class LABEL_TYPE>
class GridRagStacked2D;

template<std::size_t DIM, class LABELS_PROXY>
class GridRag;


// \cond SUPPRESS_DOXYGEN
namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<std::size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){
        typedef GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> >  GraphType;
        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef typename GraphType::NodeAdjacency NodeAdjacency;
        typedef typename GraphType::EdgeStorage EdgeStorage;

        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto labels = labelsProxy.labels();
        const auto & shape = labelsProxy.shape();

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        // assign the number of nodes to the graph
        rag.assign(numberOfLabels);

        auto makeCoord2 = [](const Coord & coord,const std::size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        if(pOpts.getActualNumThreads()<=1){
            nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
                const auto lU = labels(coord.asStdArray());
                for(std::size_t axis=0; axis<DIM; ++axis){
                    auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2.asStdArray());
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
                std::vector< container::BoostFlatSet<uint64_t> > adjacency;
            };

            std::vector<PerThread> perThreadDataVec(pOpts.getActualNumThreads());
            for(std::size_t i=0; i<perThreadDataVec.size(); ++i)
                perThreadDataVec[i].adjacency.resize(numberOfLabels);

            // collect the node-adjacency sets in parallel which needs to be merged later
            nifty::tools::parallelForEachCoordinate(threadpool, shape,[&](const int tid, const Coord & coord){
                auto & adjacency = perThreadDataVec[tid].adjacency;
                const auto lU = labels(coord.asStdArray());
                for(std::size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2.asStdArray());
                        if(lU != lV){
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });

            rag.mergeAdjacencies(perThreadDataVec, threadpool);
        }
    }
};



template<class LABELS_PROXY>
struct ComputeRag< GridRagStacked2D< LABELS_PROXY > > {

    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::BlockStorageType BlockStorageType;
    typedef GridRagStacked2D< LABELS_PROXY > RagType;
    typedef typename LABELS_PROXY::LabelType LabelType;
    typedef typename RagType::NodeAdjacency NodeAdjacency;
    typedef typename RagType::EdgeStorage EdgeStorage;


    template<class S>
    static void computeRag(
        RagType & rag,
        const S & settings
    ){
        //std::cout<<"\nphase 0\n";

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;
        typedef std::map<EdgeStorage, size_t> EdgeLengthsType;
        typedef std::vector<EdgeLengthsType> EdgeLenghtsStorage;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const LabelType numberOfLabels = labelsProxy.numberOfLabels();

        rag.assign(numberOfLabels);

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        Coord2 sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({1L, shape[1], shape[2]});

        auto & perSliceDataVec = rag.perSliceDataVec_;

        /*
         * Parallel IO version
        */
        EdgeLenghtsStorage edgeLenStorage(numberOfSlices);

        /////////////////////////////////////////////////////
        // Phase 1 : In slice node adjacency and edge count
        /////////////////////////////////////////////////////
        //std::cout<<"phase 1\n";
        {
            BlockStorageType sliceLabelsStorage(threadpool, sliceShape3, nThreads);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];
                auto & edgeLens   = edgeLenStorage[sliceIndex];

                //
                auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);

                // fetch the data for the slice
                const Coord blockBegin({sliceIndex,0L,0L});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3DView);
                auto sliceLabels = sliceLabelsFlat3DView.squeezedView();

                // do the thing
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto lU = sliceLabels(coord.asStdArray());
                    sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lU);
                    sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lU);
                    for(std::size_t axis=0; axis<2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < sliceShape2[axis]){
                            const auto lV = sliceLabels(coord2.asStdArray());
                            if(lU != lV){
                                sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lV);
                                sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lV);
                                
                                // add up the len 
                                // map insert cf.: http://stackoverflow.com/questions/97050/stdmap-insert-or-stdmap-find
                                auto e = EdgeStorage(std::min(lU,lV),std::max(lU,lV));
                                auto findEdge = edgeLens.lower_bound(e);
                                if( findEdge != edgeLens.end() && !(edgeLens.key_comp()(e, findEdge->first)) )
                                    ++(findEdge->second);
                                else
                                    edgeLens.insert(findEdge, std::make_pair(e,1));

                                if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                    ++perSliceDataVec[sliceIndex].numberOfInSliceEdges;
                                }
                            }
                        }
                    }
                });
            });
        }

        //std::cout<<"phase 2\n";
        /////////////////////////////////////////////////////
        // Phase 2 : set up the in slice edge offsets
        /////////////////////////////////////////////////////
        {
            
            for(auto sliceIndex=1; sliceIndex<numberOfSlices; ++sliceIndex){
                const auto prevOffset = perSliceDataVec[sliceIndex-1].inSliceEdgeOffset;
                const auto prevEdgeNum = perSliceDataVec[sliceIndex-1].numberOfInSliceEdges;
                perSliceDataVec[sliceIndex].inSliceEdgeOffset =  prevOffset + prevEdgeNum;

                NIFTY_CHECK_OP(perSliceDataVec[sliceIndex-1].maxInSliceNode + 1, == , perSliceDataVec[sliceIndex].minInSliceNode,
                    "unusable supervoxels for GridRagStacked2D");
            }
            const auto & lastSlice =  perSliceDataVec.back();
            rag.numberOfInSliceEdges_ = lastSlice.inSliceEdgeOffset + lastSlice.numberOfInSliceEdges;
            
        }

        //std::cout<<"phase 3\n";
        /////////////////////////////////////////////////////
        // Phase 3 : set up in slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec and edge lens
            auto & edges = rag.edges_;
            auto & edgeLengths = rag.edgeLengths_;
            edges.resize(      rag.numberOfInSliceEdges_);
            edgeLengths.resize(rag.numberOfInSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];
                const auto & edgeLensSlice = edgeLenStorage[sliceIndex];

                auto edgeIndex = sliceData.inSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;

                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v){
                            // set the edge indices in this slice
                            auto e = EdgeStorage(u, v);
                            edges[edgeIndex] = e;
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            // set the edge lens
                            edgeLengths[edgeIndex] = edgeLensSlice.at(e); 
                            // increase the edge index
                            ++edgeIndex;
                        }
                    }
                }
                NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }
        
        //std::cout<<"phase 4\n";
        /////////////////////////////////////////////////////
        // Phase 4 : between slice edges
        /////////////////////////////////////////////////////
        {

            BlockStorageType sliceAStorage(threadpool, sliceShape3, nThreads);
            BlockStorageType sliceBStorage(threadpool, sliceShape3, nThreads);
            for(auto & edgeLen : edgeLenStorage)
                edgeLen.clear();

            for(auto startIndex : {0,1}){
                parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){

                    // this seems super ugly...
                    // there must be a better way to loop in parallel
                    // over first the odd then the even coordinates
                    const auto oddIndex = bool(sliceAIndex%2);
                    if((startIndex==0 && !oddIndex) || (startIndex==1 && oddIndex )){

                        const auto sliceBIndex = sliceAIndex + 1;
                        auto & edgeLens = edgeLenStorage[sliceAIndex];

                        // fetch the data for the slice
                        const Coord beginA({sliceAIndex,0L,0L});
                        const Coord beginB({sliceBIndex,0L,0L});
                        const Coord endA({sliceAIndex+1,shape[1],shape[2]});
                        const Coord endB({sliceBIndex+1,shape[1],shape[2]});
                        
                        auto sliceAView = sliceAStorage.getView(tid);
                        auto sliceBView = sliceBStorage.getView(tid);

                        labelsProxy.readSubarray(beginA, endA, sliceAView);
                        labelsProxy.readSubarray(beginB, endB, sliceBView);
                        
                        auto sliceALabels = sliceAView.squeezedView();
                        auto sliceBLabels = sliceBView.squeezedView();

                        nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                            const auto lU = sliceALabels(coord.asStdArray());
                            const auto lV = sliceBLabels(coord.asStdArray());
                                
                            // add up the len 
                            // map insert cf.: http://stackoverflow.com/questions/97050/stdmap-insert-or-stdmap-find
                            auto e = EdgeStorage(std::min(lU,lV),std::max(lU,lV));
                            auto findEdge = edgeLens.lower_bound(e);
                            if( findEdge != edgeLens.end() && !(edgeLens.key_comp()(e, findEdge->first)) )
                                ++(findEdge->second);
                            else
                                edgeLens.insert(findEdge, std::make_pair(e,1));
                            
                            if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                ++perSliceDataVec[sliceAIndex].numberOfToNextSliceEdges;
                            }
                        });
                    }
                });
            }
        }
        
        //std::cout<<"phase 5\n";
        /////////////////////////////////////////////////////
        // Phase 5 : set up the between slice edge offsets
        /////////////////////////////////////////////////////
        {
            rag.numberOfInBetweenSliceEdges_ += perSliceDataVec[0].numberOfToNextSliceEdges;
            perSliceDataVec[0].toNextSliceEdgeOffset = rag.numberOfInSliceEdges_;
            for(auto sliceIndex=1; sliceIndex<numberOfSlices; ++sliceIndex){
                const auto prevOffset = perSliceDataVec[sliceIndex-1].toNextSliceEdgeOffset;
                const auto prevEdgeNum = perSliceDataVec[sliceIndex-1].numberOfToNextSliceEdges;
                perSliceDataVec[sliceIndex].toNextSliceEdgeOffset =  prevOffset + prevEdgeNum;

                rag.numberOfInBetweenSliceEdges_ +=  perSliceDataVec[sliceIndex].numberOfToNextSliceEdges;
            }
        }

        //std::cout<<"phase 6\n";
        /////////////////////////////////////////////////////
        // Phase 6 : set up between slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            auto & nodes = rag.nodes_;
            auto & edgeLengths = rag.edgeLengths_;
            edges.resize(      rag.numberOfInSliceEdges_ + rag.numberOfInBetweenSliceEdges_);
            edgeLengths.resize(rag.numberOfInSliceEdges_ + rag.numberOfInBetweenSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];
                const auto & edgeLensSlice = edgeLenStorage[sliceIndex];

                auto edgeIndex = sliceData.toNextSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;

                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj : rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v && v >= endNode){
                            auto e = EdgeStorage(u, v);
                            edges[edgeIndex] = e;
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  nodes[v].find(NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            // set lens
                            edgeLengths[edgeIndex] = edgeLensSlice.at(e);
                            // increase the edge index
                            ++edgeIndex;
                        }
                    }
                }
                //NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }
    }
};


} // end namespace detail_rag
// \endcond

} // end namespace graph
} // end namespace nifty
