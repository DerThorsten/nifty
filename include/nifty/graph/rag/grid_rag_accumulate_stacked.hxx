#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_STACKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_STACKED_HXX

#include <vector>
#include <cmath>

#include "nifty/graph/rag/grid_rag_accumulate.hxx"

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

namespace nifty{
namespace graph{
    
    // accumulator with data
    template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
    void accumulateEdgeFeaturesWithAccChain(

        const GridRagStacked2D<LABELS_PROXY> & rag,
        const DATA & data,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F && f,
        const AccOptions & accOptions = AccOptions()
    ){
        
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename vigra::MultiArrayShape<3>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataStorage;

        typedef array::StaticArray<int64_t,3> Coord;
        typedef array::StaticArray<int64_t,2> Coord2;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType; 
        
        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        EdgeAccChainVectorType edgeAccChainVector(rag.edgeIdUpperBound()+1);
        
        if(accOptions.setMinMax){
            parallel::parallel_foreach(threadpool, rag.edgeIdUpperBound()+1,
            [&](int tid, int edge){
                vigra::HistogramOptions histogram_opt;
                histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal); 
                edgeAccChainVector[edge].setHistogramOptions(histogram_opt);
            });
        }
        
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        const Coord2 sliceShape2({shape[1], shape[2]});
        const Coord  sliceShape3({1L,shape[1], shape[2]});
        const Coord  sliceABShape({2L,shape[1], shape[2]});
        
        const auto passesRequired = edgeAccChainVector.front().passesRequired();
        
        // do N passes of accumulator
        for(auto pass=1; pass <= passesRequired; ++pass){
            
            // in slice edges
            {
                LabelStorage labelStorage(threadpool, sliceShape3, nThreads);
                DataStorage  dataStorage(threadpool, sliceShape3, nThreads);

                parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                    
                    auto sliceLabels3DView = labelStorage.getView(tid);
                    auto sliceData3DView   = dataStorage.getView(tid);

                    // fetch the data for the slice
                    const Coord blockBegin({sliceIndex,0L,0L});
                    const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                    
                    labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabels3DView);
                    tools::readSubarray(data, blockBegin, blockEnd, sliceData3DView);
                    
                    auto sliceLabels = sliceLabels3DView.squeezedView();
                    auto sliceData   = sliceData3DView.squeezedView();

                    // do the thing 
                    nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                        
                        const auto lU = sliceLabels(coord.asStdArray());
                        const auto dataU = sliceData(coord.asStdArray());
                        VigraCoord vigraCoordU;
                        vigraCoordU[0] = sliceIndex;
                        for(size_t d=1; d<3; ++d)
                            vigraCoordU[d] = coord[d];
                        
                        for(size_t axis=0; axis<2; ++axis){
                            Coord2 coord2 = coord;
                            ++coord2[axis];
                            if(coord2[axis] < sliceShape2[axis]){
                                const auto lV = sliceLabels(coord2.asStdArray());
                                if(lU != lV){
                                    const auto dataV = sliceData(coord2.asStdArray());
                                    VigraCoord vigraCoordV;
                                    vigraCoordV[0] = sliceIndex;
                                    for(size_t d=1; d<3; ++d)
                                        vigraCoordV[d] = coord2[d];

                                    auto edge = rag.findEdge(lU,lV);
                                    edgeAccChainVector[edge].updatePassN(dataU, vigraCoordU, pass);
                                    edgeAccChainVector[edge].updatePassN(dataV, vigraCoordV, pass);
                                }
                            }
                        }
                    });
                });
            }
            
            //between slice edges
            {
                LabelStorage labelStorage(threadpool, sliceABShape, nThreads);
                DataStorage  dataStorage(threadpool,  sliceABShape, nThreads);

                for(auto startIndex : {0,1}){
                    parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){

                        // this seems super ugly...
                        // there must be a better way to loop in parallel 
                        // over first the odd then the even coordinates
                        const auto oddIndex = bool(sliceAIndex%2);
                        if((startIndex==0 && !oddIndex) || (startIndex==1 && oddIndex )){

                            const auto sliceBIndex = sliceAIndex + 1;

                            // fetch the data for the slice
                            const Coord blockABBegin({sliceAIndex,0L,0L});
                            const Coord blockABEnd(  {sliceAIndex+2, sliceShape2[0], sliceShape2[1]});

                            auto labelsAB = labelStorage.getView(tid);
                            auto dataAB   = dataStorage.getView(tid);

                            labelsProxy.readSubarray(blockABBegin, blockABEnd,  labelsAB);
                            tools::readSubarray(data,blockABBegin, blockABEnd, dataAB);

                            const Coord coordAOffset{0L,0L,0L};
                            const Coord coordBOffset{1L,0L,0L};
                            
                            auto labelsA = labelsAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                            auto labelsB = labelsAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();
                            
                            auto dataA = dataAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                            auto dataB = dataAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();

                            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                                const auto lU = labelsA(coord.asStdArray());
                                const auto lV = labelsB(coord.asStdArray());

                                const auto dataU = dataA(coord.asStdArray());
                                const auto dataV = dataB(coord.asStdArray());
                                
                                VigraCoord vigraCoordU;
                                VigraCoord vigraCoordV;
                                vigraCoordU[0] = sliceAIndex;
                                vigraCoordV[0] = sliceBIndex;
                                for(size_t d=1; d<3; ++d) {
                                    vigraCoordU[d] = coord[d];
                                    vigraCoordV[d] = coord[d];
                                }
                                    
                                auto edge = rag.findEdge(lU,lV);
                                edgeAccChainVector[edge].updatePassN(dataU, vigraCoordU, pass);
                                edgeAccChainVector[edge].updatePassN(dataV, vigraCoordV, pass);
                                    
                            });
                        }
                    });
                }
            }
        }
        f(edgeAccChainVector);
    }
    
    
    // 9 features
    template<class LABELS_PROXY, class DATA, class FEATURE_TYPE>
    void accumulateEdgeStandartFeatures(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        const DATA & data,
        const double minVal,
        const double maxVal,
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
            Quantiles         //7
        > SelectType;
        typedef acc::StandAloneAccumulatorChain<3, DataType, SelectType> AccChainType;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
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
                    for(auto qi=0; qi<7; ++qi)
                        edgeFeaturesOut(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }); 
            },
            AccOptions(minVal, maxVal)
        );
    }


} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_STACKED_HXX */
