#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "vigra/accumulator.hxx"

namespace nifty{
namespace graph{

    struct AccOptions{
        AccOptions()
        :
            setMinMax(false),
            minVal(std::numeric_limits<double>::infinity()),
            maxVal(-1.0*std::numeric_limits<double>::infinity()){
        }
        AccOptions(const double minV,
                   const double maxV)
        :
            setMinMax(true),
            minVal(minV),
            maxVal(maxV){
        }

        const bool   setMinMax;
        const double minVal;
        const double maxVal;
    };

    template<class T,class U>
    inline T 
    replaceIfNotFinite(const T & val, const U & replaceVal){
        if(std::isfinite(val))
            return val;
        else
            return replaceVal;
    }

    // accumulator with data
    template<class EDGE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType; 


        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();



        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1);
        });


        const auto passesRequired = (*perThreadEdgeAccChainVector.front()).front().passesRequired();

        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){

                vigra::HistogramOptions histogram_opt;
                histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 

                auto & edgeAccVec = *(perThreadEdgeAccChainVector[i]);
                for(auto & edgeAcc : edgeAccVec){
                    edgeAcc.setHistogramOptions(histogram_opt);
                }
            });
        }

        // do N passes of accumulator
        for(auto pass=1; pass <= passesRequired; ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord overlapBegin(0), overlapEnd(1);
            LabelBlockStorage labelsBlockStorage(threadpool, blockShape, actualNumberOfThreads);
            DataBlocKStorage dataBlocKStorage(threadpool, blockShape, actualNumberOfThreads);
            tools::parallelForEachBlockWithOverlap(threadpool,shape, blockShape, overlapBegin, overlapEnd,
            [&](
                const int tid,
                const Coord & blockCoreBegin, const Coord & blockCoreEnd,
                const Coord & blockBegin, const Coord & blockEnd
            ){

                // get the accumulator vector for this thread
                auto & accVec = *(perThreadEdgeAccChainVector[tid]);

                // actual shape of the block: might be smaller at the border as blockShape

                const auto nonOlBlockShape  = blockCoreEnd - blockCoreBegin;
                const auto actualBlockShape = blockEnd - blockBegin;

                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                auto dataBlockView = dataBlocKStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

                // loop over all coordinates in block
                nifty::tools::forEachCoordinate(nonOlBlockShape,[&](const Coord & coordU){
                    const auto lU = labelsBlockView(coordU.asStdArray());
                    for(size_t axis=0; axis<DIM; ++axis){
                        auto coordV = makeCoord2(coordU, axis);
                        if(coordV[axis] < actualBlockShape[axis]){
                            const auto lV = labelsBlockView(coordV.asStdArray());
                            if(lU != lV){
                                const auto edge = rag.findEdge(lU,lV);

                                const auto dataU = dataBlockView(coordU.asStdArray());
                                const auto dataV = dataBlockView(coordV.asStdArray());

                                
                                VigraCoord vigraCoordU;
                                VigraCoord vigraCoordV;

                                for(size_t d=0; d<DIM; ++d){
                                    vigraCoordU[d] = coordU[d]+blockBegin[d];
                                    vigraCoordV[d] = coordV[d]+blockBegin[d];
                                }

                                accVec[edge].updatePassN(dataU, vigraCoordU, pass);
                                accVec[edge].updatePassN(dataV, vigraCoordV, pass);
                            }
                        }
                    }
                });
            });
        }

        auto & resultAccVec = *(perThreadEdgeAccChainVector.front());


        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, resultAccVec.size(), 
        [&](const int tid, const int64_t edge){

            for(auto t=1; t<actualNumberOfThreads; ++t){
                resultAccVec[edge].merge((*(perThreadEdgeAccChainVector[t]))[edge]);
            }            
        });
        
        // call functor with finished acc chain
        f(resultAccVec);

        // delete 
        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadEdgeAccChainVector[i];
        });
    }

    // accumulator with data
    template<class EDGE_ACC_CHAIN, class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeAndNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef NODE_ACC_CHAIN NodeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType; 
        typedef std::vector<NodeAccChainType> NodeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();


        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);
        std::vector< NodeAccChainVectorType * > perThreadNodeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1);
            perThreadNodeAccChainVector[i] = new NodeAccChainVectorType(rag.nodeIdUpperBound()+1);
        });






        const auto numberOfEdgePasses = (*perThreadEdgeAccChainVector.front()).front().passesRequired();
        const auto numberOfNodePasses = (*perThreadNodeAccChainVector.front()).front().passesRequired();
        const auto numberOfPasses = std::max(numberOfEdgePasses, numberOfNodePasses);

        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){

                    vigra::HistogramOptions histogram_opt;
                    histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 

                    auto & edgeAccVec = *(perThreadEdgeAccChainVector[i]);
                    for(auto & edgeAcc : edgeAccVec){
                        edgeAcc.setHistogramOptions(histogram_opt);
                    }

                    auto & nodeAccVec = *(perThreadNodeAccChainVector[i]);
                    for(auto & nodeAcc : nodeAccVec){
                        nodeAcc.setHistogramOptions(histogram_opt);
                    }
            });
        }

        // do N passes of accumulator
        for(auto pass=1; pass <= numberOfPasses; ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord overlapBegin(0), overlapEnd(1);
            LabelBlockStorage labelsBlockStorage(threadpool, blockShape, actualNumberOfThreads);
            DataBlocKStorage dataBlocKStorage(threadpool, blockShape, actualNumberOfThreads);
            tools::parallelForEachBlockWithOverlap(threadpool,shape, blockShape, overlapBegin, overlapEnd,
            [&](
                const int tid,
                const Coord & blockCoreBegin, const Coord & blockCoreEnd,
                const Coord & blockBegin, const Coord & blockEnd
            ){
                // get the accumulator vector for this thread
                auto & edgeAccVec = *(perThreadEdgeAccChainVector[tid]);
                auto & nodeAccVec = *(perThreadNodeAccChainVector[tid]);
                // actual shape of the block: might be smaller at the border as blockShape

                const auto nonOlBlockShape  = blockCoreEnd - blockCoreBegin;
                const auto actualBlockShape = blockEnd - blockBegin;
 
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                auto dataBlockView = dataBlocKStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

                // loop over all coordinates in block
                nifty::tools::forEachCoordinate(nonOlBlockShape,[&](const Coord & coordU){

                    const auto lU = labelsBlockView(coordU.asStdArray());
                    const auto dataU = dataBlockView(coordU.asStdArray());
                    
                    VigraCoord vigraCoordU;
                    for(size_t d=0; d<DIM; ++d)
                        vigraCoordU[d] = coordU[d] + blockBegin[d];

                    if(pass <= numberOfNodePasses)
                        nodeAccVec[lU].updatePassN(dataU, vigraCoordU, pass);
                    
                    // accumulate the edge features
                    if(pass <= numberOfEdgePasses){
                        for(size_t axis=0; axis<DIM; ++axis){
                            auto coordV = makeCoord2(coordU, axis);
                            if(coordV[axis] < actualBlockShape[axis]){
                                const auto lV = labelsBlockView(coordV.asStdArray());
                                if(lU != lV){

                                    const auto edge = rag.findEdge(lU,lV);
                                    const auto dataV = dataBlockView(coordV.asStdArray());

                                    VigraCoord vigraCoordV;
                                    for(size_t d=0; d<DIM; ++d)
                                        vigraCoordV[d] = coordV[d] + blockBegin[d];

                                    edgeAccVec[edge].updatePassN(dataU, vigraCoordU, pass);
                                    edgeAccVec[edge].updatePassN(dataV, vigraCoordV, pass);
                                }
                            }
                        }
                    }

                });
            });
        }

        auto & edgeResultAccVec = *perThreadEdgeAccChainVector.front();
        auto & nodeResultAccVec = *perThreadNodeAccChainVector.front();

        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, edgeResultAccVec.size(), 
        [&](const int tid, const int64_t edge){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadEdgeAccChainVector[t]);
                edgeResultAccVec[edge].merge(accChainVec[edge]);           
            }
        });
        parallel::parallel_foreach(threadpool, nodeResultAccVec.size(), 
        [&](const int tid, const int64_t node){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadNodeAccChainVector[t]);
                nodeResultAccVec[node].merge(accChainVec[node]);           
            }
        });
        
        // call functor with finished acc chain
        f(edgeResultAccVec, nodeResultAccVec);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadEdgeAccChainVector[i];
            delete perThreadNodeAccChainVector[i];
        });

    }

    // accumulate without data
    template<class EDGE_ACC_CHAIN, class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class F>
    void accumulateEdgeAndNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f
    ){
      


        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef NODE_ACC_CHAIN NodeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType; 
        typedef std::vector<NodeAccChainType> NodeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();


        std::vector< EdgeAccChainVectorType * > perThreadEdgeAccChainVector(actualNumberOfThreads);
        std::vector< NodeAccChainVectorType * > perThreadNodeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadEdgeAccChainVector[i] = new EdgeAccChainVectorType(rag.edgeIdUpperBound()+1);
            perThreadNodeAccChainVector[i] = new NodeAccChainVectorType(rag.nodeIdUpperBound()+1);
        });






        const auto numberOfEdgePasses = (*perThreadEdgeAccChainVector.front()).front().passesRequired();
        const auto numberOfNodePasses = (*perThreadNodeAccChainVector.front()).front().passesRequired();
        const auto numberOfPasses = std::max(numberOfEdgePasses, numberOfNodePasses);

        // do N passes of accumulator
        for(auto pass=1; pass <= numberOfPasses; ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord overlapBegin(0), overlapEnd(1);
            LabelBlockStorage labelsBlockStorage(threadpool, blockShape, actualNumberOfThreads);
            tools::parallelForEachBlockWithOverlap(threadpool,shape, blockShape, overlapBegin, overlapEnd,
            [&](
                const int tid,
                const Coord & blockCoreBegin, const Coord & blockCoreEnd,
                const Coord & blockBegin, const Coord & blockEnd
            ){
                // get the accumulator vector for this thread
                auto & edgeAccVec = *(perThreadEdgeAccChainVector[tid]);
                auto & nodeAccVec = *(perThreadNodeAccChainVector[tid]);
                // actual shape of the block: might be smaller at the border as blockShape

                const auto nonOlBlockShape  = blockCoreEnd - blockCoreBegin;
                const auto actualBlockShape = blockEnd - blockBegin;
 
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);

                // loop over all coordinates in block
                nifty::tools::forEachCoordinate(nonOlBlockShape,[&](const Coord & coordU){

                    const auto lU = labelsBlockView(coordU.asStdArray());
                    const auto dataU = 0.0;
                    
                    VigraCoord vigraCoordU;
                    for(size_t d=0; d<DIM; ++d)
                        vigraCoordU[d] = coordU[d] + blockBegin[d];

                    if(pass <= numberOfNodePasses)
                        nodeAccVec[lU].updatePassN(dataU, vigraCoordU, pass);
                    
                    // accumulate the edge features
                    if(pass <= numberOfEdgePasses){
                        for(size_t axis=0; axis<DIM; ++axis){
                            auto coordV = makeCoord2(coordU, axis);
                            if(coordV[axis] < actualBlockShape[axis]){
                                const auto lV = labelsBlockView(coordV.asStdArray());
                                if(lU != lV){

                                    const auto edge = rag.findEdge(lU,lV);
                                    const auto dataV = 0.0;

                                    VigraCoord vigraCoordV;
                                    for(size_t d=0; d<DIM; ++d)
                                        vigraCoordV[d] = coordV[d] + blockBegin[d];

                                    edgeAccVec[edge].updatePassN(dataU, vigraCoordU, pass);
                                    edgeAccVec[edge].updatePassN(dataV, vigraCoordV, pass);
                                }
                            }
                        }
                    }

                });
            });
        }


        auto & edgeResultAccVec = *perThreadEdgeAccChainVector.front();
        auto & nodeResultAccVec = *perThreadNodeAccChainVector.front();

        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, edgeResultAccVec.size(), 
        [&](const int tid, const int64_t edge){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadEdgeAccChainVector[t]);
                edgeResultAccVec[edge].merge(accChainVec[edge]);           
            }
        });
        parallel::parallel_foreach(threadpool, nodeResultAccVec.size(), 
        [&](const int tid, const int64_t node){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadNodeAccChainVector[t]);
                nodeResultAccVec[node].merge(accChainVec[node]);           
            }
        });
        
        // call functor with finished acc chain
        f(edgeResultAccVec, nodeResultAccVec);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadEdgeAccChainVector[i];
            delete perThreadNodeAccChainVector[i];
        });

        
    }


    // accumulator with data
    template<class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef NODE_ACC_CHAIN NodeAccChainType;
        typedef std::vector<NodeAccChainType> NodeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();


        std::vector< NodeAccChainVectorType * > perThreadNodeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadNodeAccChainVector[i] = new NodeAccChainVectorType(rag.nodeIdUpperBound()+1);
        });


        const auto numberOfPasses = (*perThreadNodeAccChainVector.front()).front().passesRequired();

        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, actualNumberOfThreads,
            [&](int tid, int i){

                    vigra::HistogramOptions histogram_opt;
                    histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 

                    auto & nodeAccVec = *(perThreadNodeAccChainVector[i]);
                    for(auto & nodeAcc : nodeAccVec){
                        nodeAcc.setHistogramOptions(histogram_opt);
                    }
            });
        }

        // do N passes of accumulator
        for(auto pass=1; pass <= numberOfPasses; ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord overlapBegin(0), overlapEnd(1);
            LabelBlockStorage labelsBlockStorage(threadpool, blockShape, actualNumberOfThreads);
            DataBlocKStorage dataBlocKStorage(threadpool, blockShape, actualNumberOfThreads);
            tools::parallelForEachBlockWithOverlap(threadpool,shape, blockShape, overlapBegin, overlapEnd,
            [&](
                const int tid,
                const Coord & blockCoreBegin, const Coord & blockCoreEnd,
                const Coord & blockBegin, const Coord & blockEnd
            ){
                // get the accumulator vector for this thread
                auto & nodeAccVec = *(perThreadNodeAccChainVector[tid]);
                // actual shape of the block: might be smaller at the border as blockShape

                const auto nonOlBlockShape  = blockCoreEnd - blockCoreBegin;
                const auto actualBlockShape = blockEnd - blockBegin;
 
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                auto dataBlockView = dataBlocKStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

                // loop over all coordinates in block
                nifty::tools::forEachCoordinate(nonOlBlockShape,[&](const Coord & coordU){

                    const auto lU = labelsBlockView(coordU.asStdArray());
                    const auto dataU = dataBlockView(coordU.asStdArray());
                    
                    VigraCoord vigraCoordU;
                    for(size_t d=0; d<DIM; ++d)
                        vigraCoordU[d] = coordU[d] + blockBegin[d];

                    nodeAccVec[lU].updatePassN(dataU, vigraCoordU, pass);
                });
            });
        }

        auto & nodeResultAccVec = *perThreadNodeAccChainVector.front();

        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, nodeResultAccVec.size(), 
        [&](const int tid, const int64_t node){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadNodeAccChainVector[t]);
                nodeResultAccVec[node].merge(accChainVec[node]);           
            }
        });
        
        // call functor with finished acc chain
        f(nodeResultAccVec);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadNodeAccChainVector[i];
        });

    }


    // accumulate without data
    template<class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class F>
    void accumulateNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f
    ){
      


        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;


        typedef NODE_ACC_CHAIN NodeAccChainType;
        typedef std::vector<NodeAccChainType> NodeAccChainVectorType; 

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();

        std::vector< NodeAccChainVectorType * > perThreadNodeAccChainVector(actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            perThreadNodeAccChainVector[i] = new NodeAccChainVectorType(rag.nodeIdUpperBound()+1);
        });


        const auto numberOfPasses = (*perThreadNodeAccChainVector.front()).front().passesRequired();


        // do N passes of accumulator
        for(auto pass=1; pass <= numberOfPasses; ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord overlapBegin(0), overlapEnd(1);
            LabelBlockStorage labelsBlockStorage(threadpool, blockShape, actualNumberOfThreads);
            tools::parallelForEachBlockWithOverlap(threadpool,shape, blockShape, overlapBegin, overlapEnd,
            [&](
                const int tid,
                const Coord & blockCoreBegin, const Coord & blockCoreEnd,
                const Coord & blockBegin, const Coord & blockEnd
            ){
                // get the accumulator vector for this thread
                auto & nodeAccVec = *(perThreadNodeAccChainVector[tid]);
                // actual shape of the block: might be smaller at the border as blockShape

                const auto nonOlBlockShape  = blockCoreEnd - blockCoreBegin;
                const auto actualBlockShape = blockEnd - blockBegin;
 
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);

                // loop over all coordinates in block
                nifty::tools::forEachCoordinate(nonOlBlockShape,[&](const Coord & coordU){

                    const auto lU = labelsBlockView(coordU.asStdArray());
                    const auto dataU = 0.0;
                    
                    VigraCoord vigraCoordU;
                    for(size_t d=0; d<DIM; ++d)
                        vigraCoordU[d] = coordU[d] + blockBegin[d];

  
                    nodeAccVec[lU].updatePassN(dataU, vigraCoordU, pass);
                    
                });
            });
        }

        auto & nodeResultAccVec = *perThreadNodeAccChainVector.front();


        parallel::parallel_foreach(threadpool, nodeResultAccVec.size(), 
        [&](const int tid, const int64_t node){
            for(auto t=1; t<actualNumberOfThreads; ++t){
                auto & accChainVec = *(perThreadNodeAccChainVector[t]);
                nodeResultAccVec[node].merge(accChainVec[node]);           
            }
        });
        
        // call functor with finished acc chain
        f(nodeResultAccVec);


        parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
        [&](const int tid, const int64_t i){
            delete perThreadNodeAccChainVector[i];
        });
    }


    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateMeanAndLength(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;

        typedef FEATURE_TYPE DataType;
        typedef acc::Select< acc::DataArg<1>, acc::Mean, acc::Count> SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;


        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        //std::cout<<"Using "<<actualNumberOfThreads<<"\n";

        accumulateEdgeAndNodeFeaturesWithAccChain<AccChainType,AccChainType>(rag, data, blockShape, pOpts, threadpool,
        [&](
            const std::vector<AccChainType> & edgeAccChainVec,
            const std::vector<AccChainType> & nodeAccChainVec
        ){
            parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                const int tid, const int64_t edge
            ){
                edgeFeaturesOut(edge, 0) = acc::get<acc::Mean>(edgeAccChainVec[edge]);
                edgeFeaturesOut(edge, 1) = acc::get<acc::Count>(edgeAccChainVec[edge]);
            });

            parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                const int tid, const int64_t node
            ){
                nodeFeaturesOut(node, 0) = acc::get<acc::Mean>(nodeAccChainVec[node]);
                nodeFeaturesOut(node, 1) = acc::get<acc::Count>(nodeAccChainVec[node]);
            });
        });
    }




    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateEdgeMeanAndLength(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & out,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;

        typedef FEATURE_TYPE DataType;
        typedef acc::Select< acc::DataArg<1>, acc::Mean, acc::Count> SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> EdgeAccChainType;


        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        // allocate a ach chain vector for each thread
        accumulateEdgeFeaturesWithAccChain<EdgeAccChainType>(rag, data, blockShape, pOpts, threadpool,
        [&](
            const std::vector<EdgeAccChainType> & accChainVec
        ){
            parallel::parallel_foreach(threadpool, accChainVec.size(),[&](
                const int tid, const int64_t edge
            ){
                out(edge, 0) = acc::get<acc::Mean>(accChainVec[edge]);
                out(edge, 1) = acc::get<acc::Count>(accChainVec[edge]);
            });
        });
    }


    // 11 features
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateStandartFeatures(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const double minVal,
        const double maxVal,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;


        typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;


        typedef acc::Select< 
            acc::DataArg<1>, 
            acc::Mean,        //1
            acc::Variance,    //1
            acc::Skewness,    //1
            acc::Kurtosis,    //1
            Quantiles         //7
        > SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;


        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        accumulateEdgeAndNodeFeaturesWithAccChain<AccChainType,AccChainType>(
            rag, 
            data, 
            blockShape, 
            pOpts, 
            threadpool,
            [&](
                const std::vector<AccChainType> & edgeAccChainVec,
                const std::vector<AccChainType> & nodeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    edgeFeaturesOut(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    edgeFeaturesOut(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    edgeFeaturesOut(edge, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    edgeFeaturesOut(edge, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        edgeFeaturesOut(edge, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                });
              
                parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                    const int tid, const int64_t node
                ){
                    const auto & chain = nodeAccChainVec[node];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    nodeFeaturesOut(node, 0) = replaceIfNotFinite(mean,     0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    nodeFeaturesOut(node, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    nodeFeaturesOut(node, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi){
                        nodeFeaturesOut(node, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                });

            },AccOptions(minVal, maxVal)
        );
    }


    // 11 features
    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateEdgeStandartFeatures(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const double minVal,
        const double maxVal,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;


        typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;


        typedef acc::Select<
            acc::DataArg<1>,
            acc::Mean,        //1
            acc::Variance,    //1
            acc::Skewness,    //1
            acc::Kurtosis,    //1
            Quantiles         //7
        > SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            blockShape,
            pOpts,
            threadpool,
            [&](
                const std::vector<AccChainType> & edgeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    edgeFeaturesOut(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    edgeFeaturesOut(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    edgeFeaturesOut(edge, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    edgeFeaturesOut(edge, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        edgeFeaturesOut(edge, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }); 
            },
            AccOptions(minVal, maxVal)
        );

    }


    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateNodeStandartFeatures(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const double minVal,
        const double maxVal,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;


        typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
        typedef acc::StandardQuantiles<SomeHistogram > Quantiles;


        typedef acc::Select< 
            acc::DataArg<1>, 
            acc::Mean,        //1
            acc::Variance,    //1
            acc::Skewness,    //1
            acc::Kurtosis,    //1
            Quantiles         //7
        > SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;


        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        accumulateNodeFeaturesWithAccChain<AccChainType>(
            rag, 
            data, 
            blockShape, 
            pOpts, 
            threadpool,
            [&](
                const std::vector<AccChainType> & nodeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                    const int tid, const int64_t node
                ){
                    const auto & chain = nodeAccChainVec[node];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    nodeFeaturesOut(node, 0) = replaceIfNotFinite(mean,     0.0);
                    nodeFeaturesOut(node, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    nodeFeaturesOut(node, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                    nodeFeaturesOut(node, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi){
                        nodeFeaturesOut(node, 4+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                });

            },
            AccOptions(minVal, maxVal)
        );
    }

    // number of features = 1 + 3*DIM
    template<size_t DIM, class LABELS_PROXY, class FEATURE_TYPE>
    void accumulateGeometricNodeFeatures(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & nodeFeaturesOut,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;



        typedef acc::Coord<acc::Principal<acc::CoordinateSystem> >                 RegionAxes;
        typedef acc::Select< 
            acc::DataArg<1>, 
            acc::Count,
            acc::RegionCenter,
            RegionAxes
        > SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, float, SelectType> AccChainType;


        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        accumulateNodeFeaturesWithAccChain<AccChainType>(
            rag, 
            blockShape, 
            pOpts, 
            threadpool,
            [&](
                const std::vector<AccChainType> & nodeAccChainVec
            ){
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, nodeAccChainVec.size(),[&](
                    const int tid, const int64_t node
                ){
                    const auto & chain = nodeAccChainVec[node];
                    
                    const auto regionCenter = get<acc::RegionCenter>(chain);
                    const auto regionAxes = get<RegionAxes>(chain);

                    nodeFeaturesOut(node, 0) = get<acc::Count>(chain);

                    for(auto d=0; d<DIM; ++d){
                        nodeFeaturesOut(node, 1+d) = regionCenter[d];
                    }
                    for(auto d=0; d<DIM; ++d){
                        nodeFeaturesOut(node, 1+DIM+d) = regionAxes[d];
                    }
                    for(auto d=0; d<DIM; ++d){
                        nodeFeaturesOut(node, 1+2*DIM+d) = regionAxes[d+DIM];
                    }

                });

            }
        );
    }


    /**
     * @brief      accumulate geometric features
     *
     * @param[in]  rag              The rag
     * @param[in]  blockShape       The block shape
     * @param      edgeFeaturesOut  The edge features out
     * @param[in]  numberOfThreads  The number of threads
     *
     * @tparam     DIM              Dimension of the rag
     * @tparam     LABELS_PROXY     Label Proxy type of the rag
     * @tparam     FEATURE_TYPE     OutType of the features
     * 
     * @detail 
     *   
     *   feature 0 : mean edge length
     * 
     * 
     */
    template<size_t DIM, class LABELS_PROXY, class FEATURE_TYPE>
    void accumulateGeometricEdgeFeatures(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const array::StaticArray<int64_t, DIM> & blockShape,
        marray::View<FEATURE_TYPE> & edgeFeaturesOut,
        const int numberOfThreads = -1
    ){

        namespace acc = vigra::acc;
        typedef FEATURE_TYPE DataType;


        //typedef Coord<RootDivideByCount<Principal<PowerSum<2> > > > RegionRadii;
        typedef acc::Coord<acc::Principal<acc::CoordinateSystem> >                 RegionAxes;
        typedef acc::Principal<acc::CoordinateSystem>   PrincipalAxes;

        typedef acc::Select<  
            acc::DataArg<1>,
            acc::Count,
            //,
            //PrincipalAxes   
            acc::RegionCenter,
            //RegionRadii,
            RegionAxes
            //PrincipalAxes
        > SelectType;


        
        
        
        
        
        
        
        typedef acc::StandAloneAccumulatorChain<DIM,float, SelectType> AccChainType;

        
        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        
        accumulateEdgeAndNodeFeaturesWithAccChain<AccChainType,AccChainType>(
            rag, 
            blockShape, 
            pOpts, 
            threadpool,
            [&](
                const std::vector<AccChainType> & edgeAccChainVec,
                const std::vector<AccChainType> & nodeAccChainVec
            ){
                
                using namespace vigra::acc;

                parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
                    const int tid, const int64_t edge
                ){
                    const auto uv = rag.uv(edge);
                    const auto & chainE = edgeAccChainVec[edge];
                    const auto & chainU = nodeAccChainVec[uv.first];
                    const auto & chainV = nodeAccChainVec[uv.second];

                    const float countE = get<acc::Count>(chainE);
                    const float countU = get<acc::Count>(chainU);
                    const float countV = get<acc::Count>(chainV);

                    const auto pAxesE = get<RegionAxes>(chainE);
                    const auto pAxesU = get<RegionAxes>(chainU);
                    const auto pAxesV = get<RegionAxes>(chainV);

                    
                    const auto rCenterE = get<acc::RegionCenter>(chainE);
                    const auto rCenterU = get<acc::RegionCenter>(chainU);
                    const auto rCenterV = get<acc::RegionCenter>(chainV);


                    // count based
                    edgeFeaturesOut(edge, 0) = countE;
                    edgeFeaturesOut(edge, 1) = std::abs(countU - countV);
                    edgeFeaturesOut(edge, 2) = countU + countV;
                    edgeFeaturesOut(edge, 3) = countU * countV;
                    edgeFeaturesOut(edge, 4) = std::min(countU , countV);
                    edgeFeaturesOut(edge, 5) = std::max(countU , countV);
                    edgeFeaturesOut(edge, 6) = countE / std::min(countU , countV);
                    edgeFeaturesOut(edge, 7) = countE / std::max(countU , countV);
                    edgeFeaturesOut(edge, 8) = countE / (countU + countV);


                    // region center based
                    const auto dEU = rCenterE-rCenterU;
                    const auto dEV = rCenterE-rCenterV;
                    const auto ndEU = vigra::norm(dEU);
                    const auto ndEV = vigra::norm(dEV);

                    const auto cosOfAngle =  vigra::dot(dEU,dEV)/(ndEU*ndEV);
                    const auto angle = std::acos(cosOfAngle);

                    edgeFeaturesOut(edge, 9)  = std::abs(ndEU - ndEV);
                    edgeFeaturesOut(edge, 10) = ndEU + ndEV;
                    edgeFeaturesOut(edge, 11) = ndEU * ndEV;
                    edgeFeaturesOut(edge, 12) = std::min(ndEU , ndEV);
                    edgeFeaturesOut(edge, 13) = std::max(ndEU , ndEV);
                    edgeFeaturesOut(edge, 14) =  replaceIfNotFinite(angle, 3.14159265358979323846);


                    // region axis based
                    vigra::TinyVector<float,DIM> r0U, r0V, r0E;

                    for(size_t i=0; i<DIM; ++i){
                        r0U = pAxesU[i];
                        r0V = pAxesV[i];
                        r0E = pAxesE[i];
                    }

                    // angle UV 
                    const auto aUV=std::acos(vigra::dot(r0U,r0V)/(vigra::norm(r0U),vigra::norm(r0V)));
                    // angle UE 
                    const auto aUE=std::acos(vigra::dot(r0U,r0E)/(vigra::norm(r0U),vigra::norm(r0E)));
                    // angle VE
                    const auto aVE=std::acos(vigra::dot(r0V,r0E)/(vigra::norm(r0V),vigra::norm(r0E)));

                    edgeFeaturesOut(edge, 15) = replaceIfNotFinite(aUV, 0.0);
                    edgeFeaturesOut(edge, 16) = replaceIfNotFinite(std::abs(aUE-aVE), 0.0);


        

                });
                
            }
        );
    }










        

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX */
