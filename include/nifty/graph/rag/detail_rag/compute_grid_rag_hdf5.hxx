#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX

#include <vector>



#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class Hdf5Labels;

template<size_t DIM, class LABELS_PROXY>
class GridRag;

template<class LABEL_TYPE>
class GridRagStacked2D;


namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  Hdf5Labels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){


        typedef array::StaticArray<int64_t, DIM> Coord;


        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        Coord blockShape,blockShapeWithBorder;
        for(auto d=0; d<DIM; ++d){
            blockShape[d] = std::min(settings.blockShape[d], shape[d]);
            blockShapeWithBorder[d] = std::min(blockShape[d]+1, shape[d]);
        }
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto blocksPerAxis = shape/blockShape;
        
        rag.assign(numberOfLabels);


        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        //std::cout<<"acutal n threads "<<nThreads<<"\n";
        
        auto getBlockRange = [&](const Coord & blockCoord,Coord & blockBegin,
                                Coord & blockEnd, Coord & actualBlockShape){
            for(auto d=0; d<DIM; ++d){
                blockBegin[d] = blockCoord[d] * blockShape[d];
                blockEnd[d] =   std::min(shape[d], ((blockCoord[d] + 1) * blockShape[d]) +1 );
                actualBlockShape[d] = blockEnd[d] - blockBegin[d];
            }
        };



        // allocate / create data for each thread
        struct PerThreadData{
            marray::Marray<LABEL_TYPE> blockLabels;
            std::vector< container::BoostFlatSet<uint64_t> > adjacency;
        };
        std::vector<PerThreadData> perThreadDataVec(nThreads);

        parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
            perThreadDataVec[i].blockLabels.resize(blockShapeWithBorder.begin(), blockShapeWithBorder.end());
            perThreadDataVec[i].adjacency.resize(numberOfLabels);
        });
        
        Coord zeroCoord(0);
        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        //std::cout<<"settings.blockShape  "<<settings.blockShape<<"\n";
        //std::cout<<"blockShapeWithBorder "<<blockShapeWithBorder<<"\n";
        //std::cout<<"blocks per axis      "<<blocksPerAxis<<"\n";

        nifty::tools::parallelForEachCoordinate(threadpool, blocksPerAxis,
        [&](const int tid, const Coord & blockCoord){

            //std::cout<<"TID "<<tid<<"\n";
            // get begin end end of the block with an overlap of 1 
            Coord blockBegin, blockEnd, actualBlockShape;
            getBlockRange(blockCoord, blockBegin, blockEnd, actualBlockShape);

            auto blockLabels = perThreadDataVec[tid].blockLabels.view(zeroCoord.begin(), actualBlockShape.begin());

            Coord marrayShape;
            Coord viewShape;

            for(auto d=0; d<DIM; ++d){
                marrayShape[d] = perThreadDataVec[tid].blockLabels.shape(d);
                viewShape[d] = blockLabels.shape(d);
            }

            //std::cout<<"marrayShape      "<<marrayShape<<"\n";
            //std::cout<<"viewShape        "<<viewShape<<"\n";

            //std::cout<<"blockBegin       "<<blockBegin<<"\n";
            //std::cout<<"blockEnd         "<<blockEnd<<"\n";
            //std::cout<<"actualBlockShape "<<actualBlockShape<<"\n";

            //marray::Marray<LABEL_TYPE> buffer(actualBlockShape.begin(), actualBlockShape.end());

            // get the labels block from hdf5 
            // 
            ////std::cout<<"code readSubarray\n";
            labelsProxy.readSubarray(blockBegin, blockEnd, blockLabels);
            ////std::cout<<"done code readSubarray\n";
            ////std::cout<<"buffer "<<buffer.asString()<<"\n";

            // get the adjacency for each thread on its own
            auto & adjacency = perThreadDataVec[tid].adjacency;

            nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coord){
                const auto lU = blockLabels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < actualBlockShape[axis]){
                        const auto lV = blockLabels(coord2.asStdArray());
                        //std::cout<<"lU "<<lU<<" lV"<<lV<<"\n";
                        if(lU != lV){
                            //std::cout<<"HUTUHUT\n";
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });
        });
        
        for(auto perThreadData : perThreadDataVec){
            for(const auto adjacencySet : perThreadData.adjacency){
                //std::cout<<"adj size "<<adjacencySet.size()<<"\n";
            }
        }
        rag.mergeAdjacencies(perThreadDataVec, threadpool);

    }
};

/*
template<class LABEL_TYPE>
struct ComputeRag< GridRagStacked2D< Hdf5Labels<3, LABEL_TYPE> > > {

    typedef GridRagStacked2D< Hdf5Labels<3, LABEL_TYPE> > RagType;

    template<class S>
    static void computeRag(
        RagType & rag,
        const S & settings
    ){
        std::cout<<"\nphase 0\n";

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const LABEL_TYPE numberOfLabels = labelsProxy.numberOfLabels();
        
        rag.assign(numberOfLabels);

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});

        auto & perSliceDataVec = rag.perSliceDataVec_;

        /////////////////////////////////////////////////////
        // Phase 1 : In slice node adjacency and edge count
        /////////////////////////////////////////////////////
        std::cout<<"phase 1\n";
        { 
            // allocate / create data for each thread
            struct PerThreadData{
               marray::Marray<LABEL_TYPE> sliceLabels;
            };
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceLabels.resize(sliceShape3.begin(), sliceShape3.end());
            });

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                // fetch the data for the slice
                auto & sliceLabelsFlat3D = perThreadDataVec[tid].sliceLabels;
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3D);
                auto sliceLabels = sliceLabelsFlat3D.squeezedView();

                // do the thing 
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto lU = sliceLabels(coord.asStdArray());
                    sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lU);
                    sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lU);
                    for(size_t axis=0; axis<2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < sliceShape2[axis]){
                            const auto lV = sliceLabels(coord2.asStdArray());
                            if(lU != lV){
                                sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lV);
                                sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lV);
                                if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                    ++perSliceDataVec[sliceIndex].numberOfInSliceEdges;
                                }
                            }
                        }
                    }
                });
            });
        }

        std::cout<<"phase 2\n";
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

        std::cout<<"phase 3\n";
        /////////////////////////////////////////////////////
        // Phase 3 : set up in slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.inSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            ++edgeIndex;
                        }
                    }
                }
                NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }

        std::cout<<"phase 4\n";
        /////////////////////////////////////////////////////
        // Phase 4 : between slice edges
        /////////////////////////////////////////////////////
        {

            // allocate / create data for each thread
            struct PerThreadData{
               marray::Marray<LABEL_TYPE> sliceAB;
            };
            Coord sliceABShape({int64_t(2),shape[1], shape[2]});
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceAB.resize(sliceABShape.begin(), sliceABShape.end());
            });

            for(auto startIndex : {0,1}){
                parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){

                    // this seems super ugly...
                    // there must be a better way to loop in parallel 
                    // over first the odd then the even coordinates
                    const auto oddIndex = bool(sliceAIndex%2);
                    if(startIndex==0 && !oddIndex || startIndex==1 && oddIndex ){

                        const auto sliceBIndex = sliceAIndex + 1;

                        // fetch the data for the slice
                        const Coord blockABBegin({sliceAIndex,int64_t(0),int64_t(0)});
                        const Coord blockABEnd({sliceAIndex+2, sliceShape2[0], sliceShape2[1]});
                        auto & sliceAB  = perThreadDataVec[tid].sliceAB;
                        labelsProxy.readSubarray(blockABBegin, blockABEnd, sliceAB);
                        const Coord coordAOffset{0L,0L,0L};
                        const Coord coordBOffset{1L,0L,0L};
                        auto sliceALabels = sliceAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                        auto sliceBLabels = sliceAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();

                        nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                            const auto lU = sliceALabels(coord.asStdArray());
                            const auto lV = sliceBLabels(coord.asStdArray());
                            if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                ++perSliceDataVec[sliceAIndex].numberOfToNextSliceEdges;
                            }                      
                        });
                    }
                });
            }
        }

        std::cout<<"phase 5\n";
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
            const auto & lastSlice =  perSliceDataVec.back();
        }

        std::cout<<"phase 6\n";
        /////////////////////////////////////////////////////
        // Phase 6 : set up between slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_ + rag.numberOfInBetweenSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.toNextSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v && v >= endNode){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            ++edgeIndex;
                        }
                    }
                }
                //NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }
    } 
};
*/




template<class LABELS_PROXY>
struct ComputeRag< GridRagStacked2D< LABELS_PROXY > > {

    typedef GridRagStacked2D< LABELS_PROXY > RagType;
    typedef typename LABELS_PROXY::LabelType LabelType;
    
    template<class S>
    static void computeRag(
        RagType & rag,
        const S & settings
    ){
        std::cout<<"\nphase 0\n";

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const LabelType numberOfLabels = labelsProxy.numberOfLabels();
        
        rag.assign(numberOfLabels);

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});

        auto & perSliceDataVec = rag.perSliceDataVec_;

        /////////////////////////////////////////////////////
        // Phase 1 : In slice node adjacency and edge count
        /////////////////////////////////////////////////////
        std::cout<<"phase 1\n";
        { 
            // allocate / create data for each thread
            struct PerThreadData{
               marray::Marray<LabelType> sliceLabels;
            };
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceLabels.resize(sliceShape3.begin(), sliceShape3.end());
            });

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                // fetch the data for the slice
                auto & sliceLabelsFlat3D = perThreadDataVec[tid].sliceLabels;
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3D);
                auto sliceLabels = sliceLabelsFlat3D.squeezedView();

                // do the thing 
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto lU = sliceLabels(coord.asStdArray());
                    sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lU);
                    sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lU);
                    for(size_t axis=0; axis<2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < sliceShape2[axis]){
                            const auto lV = sliceLabels(coord2.asStdArray());
                            if(lU != lV){
                                sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lV);
                                sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lV);
                                if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                    ++perSliceDataVec[sliceIndex].numberOfInSliceEdges;
                                }
                            }
                        }
                    }
                });
            });
        }

        std::cout<<"phase 2\n";
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

        std::cout<<"phase 3\n";
        /////////////////////////////////////////////////////
        // Phase 3 : set up in slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.inSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            ++edgeIndex;
                        }
                    }
                }
                NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }

        std::cout<<"phase 4\n";
        /////////////////////////////////////////////////////
        // Phase 4 : between slice edges
        /////////////////////////////////////////////////////
        {

            // allocate / create data for each thread
            struct PerThreadData{
               marray::Marray<LabelType> sliceAB;
            };
            Coord sliceABShape({int64_t(2),shape[1], shape[2]});
            std::vector<PerThreadData> perThreadDataVec(nThreads);
            parallel::parallel_foreach(threadpool, nThreads, [&](const int tid, const int i){
                perThreadDataVec[i].sliceAB.resize(sliceABShape.begin(), sliceABShape.end());
            });

            for(auto startIndex : {0,1}){
                parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){

                    // this seems super ugly...
                    // there must be a better way to loop in parallel 
                    // over first the odd then the even coordinates
                    const auto oddIndex = bool(sliceAIndex%2);
                    if(startIndex==0 && !oddIndex || startIndex==1 && oddIndex ){

                        const auto sliceBIndex = sliceAIndex + 1;

                        // fetch the data for the slice
                        const Coord blockABBegin({sliceAIndex,int64_t(0),int64_t(0)});
                        const Coord blockABEnd({sliceAIndex+2, sliceShape2[0], sliceShape2[1]});
                        auto & sliceAB  = perThreadDataVec[tid].sliceAB;
                        labelsProxy.readSubarray(blockABBegin, blockABEnd, sliceAB);
                        const Coord coordAOffset{0L,0L,0L};
                        const Coord coordBOffset{1L,0L,0L};
                        auto sliceALabels = sliceAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                        auto sliceBLabels = sliceAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();

                        nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                            const auto lU = sliceALabels(coord.asStdArray());
                            const auto lV = sliceBLabels(coord.asStdArray());
                            if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                ++perSliceDataVec[sliceAIndex].numberOfToNextSliceEdges;
                            }                      
                        });
                    }
                });
            }
        }

        std::cout<<"phase 5\n";
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
            const auto & lastSlice =  perSliceDataVec.back();
        }

        std::cout<<"phase 6\n";
        /////////////////////////////////////////////////////
        // Phase 6 : set up between slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_ + rag.numberOfInBetweenSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.toNextSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v && v >= endNode){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
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
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HDF5_HXX */
