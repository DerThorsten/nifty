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



    template<class ACC_CHAIN, size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    std::vector<ACC_CHAIN> accumulateWithAccChain(

        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        marray::View<FEATURE_TYPE> & out,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<DIM>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataBlocKStorage;

        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef ACC_CHAIN AccChainType;
        typedef std::vector<AccChainType> AccChainVectorType; 


        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        const auto & shape = rag.shape();


        std::vector< AccChainVectorType > perThreadAccChainVector(actualNumberOfThreads, 
            AccChainVectorType(rag.edgeIdUpperBound()));


        // do N passes of accumulator

        for(auto pass=1; pass<= perThreadAccChainVector.front().front().passesRequired(); ++pass){

            // LOOP IN PARALLEL OVER ALL BLOCKS WITH A CERTAIN OVERLAP
            const Coord blockShape(100), overlapBegin(0), overlapEnd(1);
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
                const auto actualBlockShape = blockEnd - blockBegin;
                // read the labels block and the data block
                auto labelsBlockView = labelsBlockStorage.getView(actualBlockShape, tid);
                auto dataBlockView = dataBlocKStorage.getView(actualBlockShape, tid);
                tools::readSubarray(rag.labelsProxy(), blockBegin, blockEnd, labelsBlockView);
                tools::readSubarray(data, blockBegin, blockEnd, dataBlockView);

                // loop over all coordinates in block
                 nifty::tools::forEachCoordinate(actualBlockShape,[&](const Coord & coordU){
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
                                accVec[edge].updatePassN(dataU, vigraCoordV, pass);
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
        return resultAccVec;
    }


    template<size_t DIM, class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateEdgeMeanAndLength(
        const GridRag<DIM, LABELS_PROXY> & rag,
        const DATA & data,
        marray::View<FEATURE_TYPE> & out,
        const int numberOfThreads = -1
    ){
        namespace acc = vigra::acc;

        typedef FEATURE_TYPE DataType;
        typedef acc::Select< acc::DataArg<1>, acc::Mean, acc::Count> SelectType;
        typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChain;


        // threadpool

        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();


        // allocate a ach chain vector for each thread
        const auto accChainVec = accumulateWithAccChain<AccChain>(rag, data, out, pOpts, threadpool);


        parallel::parallel_foreach(threadpool, accChainVec.size(),[&](
            const int tid, const int64_t edge
        ){
            out(edge, 0) = acc::get<acc::Mean>(accChainVec[edge]);
            out(edge, 1) = acc::get<acc::Mean>(accChainVec[edge]);
        });

    }


    

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_HXX */
