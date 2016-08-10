#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX

#include <vector>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "vigra/accumulator.hxx"

namespace nifty{
namespace graph{



    template<class EDGE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> AccChainVectorType; 


        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();


        std::vector< AccChainVectorType > perThreadAccChainVector(actualNumberOfThreads, 
            AccChainVectorType(rag.edgeIdUpperBound()+1));

        // do N passes of accumulator
        for(auto pass=1; pass <= perThreadAccChainVector.front().front().passesRequired(); ++pass){

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
                auto & accVec = perThreadAccChainVector[tid];

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

                                const auto dataU = labelsBlockView(coordU.asStdArray());
                                const auto dataV = labelsBlockView(coordV.asStdArray());

                                
                                VigraCoord vigraCoordU;
                                VigraCoord vigraCoordV;

                                for(size_t d=0; d<DIM; ++d){
                                    vigraCoordU[d] = coordU[d];
                                    vigraCoordV[d] = coordV[d];
                                }

                                accVec[edge].updatePassN(dataU, vigraCoordU, pass);
                                accVec[edge].updatePassN(dataV, vigraCoordV, pass);
                            }
                        }
                    }
                });
            });
        }

        auto & resultAccVec = perThreadAccChainVector.front();


        // merge the accumulators parallel
        parallel::parallel_foreach(threadpool, resultAccVec.size(), 
        [&](const int tid, const int64_t edge){

            for(auto t=1; t<actualNumberOfThreads; ++t){
                resultAccVec[edge].merge(perThreadAccChainVector[t][edge]);
            }            
        });
        
        // call functor with finished acc chain
        f(resultAccVec);
    }


    template<class EDGE_ACC_CHAIN, class NODE_ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeAndNodeFeaturesWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        const array::StaticArray<int64_t, DIM> & blockShape,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;

        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef EDGE_ACC_CHAIN NodeAccChainType;
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


        

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX */
