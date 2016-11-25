#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_STACKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_ACCUMULATE_FILTERS_STACKED_HXX

#include <vector>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_block.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "vigra/accumulator.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

namespace nifty{
namespace graph{


// TODO implement calculations for affinity maps
// xy-edges: accumulate over max(x-channel, y-channel)
// z-edges:  accumulate over z-channel

template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
void accumulateEdgeFeaturesFromFiltersWithAccChain(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    const bool keepXYOnly,
    const bool keepZOnly,
    const parallel::ParallelOptions & pOpts,
    parallel::ThreadPool & threadpool,
    F && f
){
    typedef LABELS_PROXY LabelsProxyType;
    typedef typename DATA::DataType DataType;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    
    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<float> FilterBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;
    
    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType; 
    typedef std::vector<AccChainVectorType> ChannelAccChainVectorType; 

    const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();
    
    // filters: TODO make accessible
    features::GaussianSmoothing gs;
    features::LaplacianOfGaussian log;
    features::HessianOfGaussianEigenvalues hog;

    std::vector<features::FilterBase*> filters({&gs, &log, &hog});
    // sigmas: TODO make accessible
    std::vector<double> sigmas({1.6,4.2,8.2});

    features::ApplyFilters<2> applyFilters(sigmas, filters);
    size_t numberOfChannels = applyFilters.numberOfChannels();
    
    //ChannelAccChainVectorType channelAccChainVector( rag.edgeIdUpperBound()+1, 
    //    AccChainVectorType(numberOfChannels) );

    uint64_t numberOfSlices = shape[0];
    
    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1L,shape[1], shape[2]});
    Coord filterShape({int64_t(numberOfChannels), shape[1], shape[2]});
    
    // filter computation and accumulation
    // FIXME we only support 1 pass for now
    //for(auto pass = 1; pass <= channelAccChainVector.front().front().passesRequired(); ++pass) {
    int pass = 1;
    {

        // edge acc vectors for multiple threads
        // FIXME why do we use pointers here
        std::vector< ChannelAccChainVectorType * > perThreadChannelAccChainVector(actualNumberOfThreads);

        LabelBlockStorage labelsAStorage(threadpool, sliceShape3, 1);
        LabelBlockStorage labelsBStorage(threadpool, sliceShape3, 1);
        DataBlockStorage dataAStorage(threadpool, sliceShape3, 1);
        DataBlockStorage dataBStorage(threadpool, sliceShape3, 1);
        FilterBlockStorage filterAStorage(threadpool, filterShape, 1);
        FilterBlockStorage filterBStorage(threadpool, filterShape, 1);

        // process slice 0
        Coord beginA({int64_t(0),int64_t(0),int64_t(0)}); 
        Coord endA({int64_t(1),shape[1],shape[2]}); 
        auto labelsA = labelsAStorage.getView(0);  
        labelsProxy.readSubarray(beginA, endA, labelsA);
        auto labelsASqueezed = labelsA.squeezedView();
            
        auto dataA = dataAStorage.getView(0);
        tools::readSubarray(data, beginA, endA, dataA);
        auto dataASqueezed = dataA.squeezedView();

        marray::Marray<float> dataCopy; // in case we need to copy data for non-float type
        marray::View<float> dataAView;

        if( typeid(DataType) == typeid(float) ) {
            dataAView = dataASqueezed;
        }
        else {
            dataCopy.resize(sliceShape2.begin(),sliceShape2.end());
            std::copy(dataASqueezed.begin(), dataASqueezed.end(), dataCopy.begin());
            dataAView.assign(sliceShape2.begin(), sliceShape2.end(), &dataCopy(0) );
        }

        auto filterA = filterAStorage.getView(0);

        applyFilters(dataAView, filterA, threadpool);
    
        Coord beginB;
        Coord endB;

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);

        for(uint64_t sliceId = 0; sliceId < numberOfSlices; ++sliceId) {

            std::cout << sliceId << " / " << numberOfSlices << std::endl;
            
            // set the correct histogram for each filter from 0th slice
            if(sliceId == 0){
                Coord cShape({1L,sliceShape2[0],sliceShape2[1]});
                parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
                    auto & histoOpts = histoOptionsVec[c];
                    
                    Coord cBegin({c,0L,0L});
                    auto channelView = filterA.view(cBegin.begin(), cShape.begin());
                    auto minMax = std::minmax_element(channelView.begin(), channelView.end());
                    auto min = *(minMax.first); // filterA[:,:,:,c].min()
                    auto max = *(minMax.second); // filterA[:,:,:,c].max()
                    histoOpts.setMinMax(min,max);
                });
                //parallel::parallel_foreach(threadpool, rag.edgeIdUpperBound() + 1, [&](const int tid, const int64_t edge){
                //    auto & edgeAccChainVec = channelAccChainVector[edge];
                //    for(int c = 0; c < numberOfChannels; ++c)
                //        edgeAccChainVec[c].setHistogramOptions(histoOptionsVec[c]);
                //});
            }


            if( !keepZOnly ) {
                // resize the channel acc chain thread vector
                // FIXME why do we use pointers / allocate dynamically here ?
                // I don't know if this makes sense in the setting we have here, discuss with him!
                parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
                [&](const int tid, const int64_t i){
                    perThreadChannelAccChainVector[i] = new ChannelAccChainVectorType( rag.numberOfInSliceEdges(sliceId),
                        AccChainVectorType(numberOfChannels) );
                    // set minmax
                    auto & perThreadEdgeAccChainVector = *(perThreadChannelAccChainVector[i]);
                    for(int64_t edge = 0; edge < rag.numberOfInSliceEdges(sliceId); ++edge){
                        for(int c = 0; c < numberOfChannels; ++c)
                            perThreadEdgeAccChainVector[edge][c].setHistogramOptions(histoOptionsVec[c]);
                    }
                });
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceId);

                // accumulate filter for the inner slice edges
                nifty::tools::parallelForEachCoordinate(threadpool, sliceShape2, [&](const int tid, const Coord2 coord){

                    auto & channelAccChainVec = *(perThreadChannelAccChainVector[tid]);
                    const auto lU = labelsASqueezed(coord.asStdArray());
                    for(int axis = 0; axis < 2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if( coord2[axis] < sliceShape2[axis]) {
                            const auto lV = labelsASqueezed(coord2.asStdArray());
                            if(lU != lV) {
                                // FIXME I really don't get this vigra coord buisness -> ask thorsten...
                                // do we need 0 or the slice coordinate as 0 entry ???
                                VigraCoord vigraCoordU;    
                                VigraCoord vigraCoordV;    
                                vigraCoordU[0] = sliceId;
                                vigraCoordV[0] = sliceId;
                                for(int d = 1; d < 3; ++d){
                                    vigraCoordU[d] = coord[d-1];
                                    vigraCoordV[d] = coord2[d-1];
                                }

                                const auto edge = rag.findEdge(lU,lV) - inEdgeOffset;
                                
                                for(int c = 0; c < numberOfChannels; ++c) {
                                    const auto fU = filterA(c, coord[0], coord[1]);
                                    const auto fV = filterA(c, coord2[0], coord2[1]);
                                    channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                                    channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
                                }
                            }
                        }
                    }
                });
                
                // merge
                parallel::parallel_foreach(threadpool, rag.numberOfInSliceEdges(sliceId), 
                [&](const int tid, const int64_t edge){
                    auto & accChainVector = *perThreadChannelAccChainVector[0];
                    for(auto t=1; t<actualNumberOfThreads; ++t){
                        auto & perThreadAccChainVec = *(perThreadChannelAccChainVector[t]);
                        for(int c = 0; c < numberOfChannels; ++c)
                            accChainVector[edge][c].merge(perThreadAccChainVec[edge][c]);
                    }            
                });

                f(*perThreadChannelAccChainVector[0], inEdgeOffset);
                
                // delete
                parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
                [&](const int tid, const int64_t i){
                    delete perThreadChannelAccChainVector[i];
                });
            }
            
            if(sliceId < numberOfSlices - 1 && keepXYOnly) {
                
                beginA = Coord({int64_t(sliceId+1),int64_t(0),int64_t(0)});
                endA = Coord({int64_t(sliceId+2),shape[1],shape[2]});
                
                auto labelsA = labelsAStorage.getView(0);  
                labelsProxy.readSubarray(beginA, endA, labelsA);
                auto labelsASqueezed = labelsA.squeezedView();
        
                auto dataA = dataAStorage.getView(0);
                tools::readSubarray(data, beginA, endA, dataA);
                auto dataASqueezed = dataA.squeezedView();

                marray::View<float> dataAView;
                if( typeid(DataType) == typeid(float) ) {
                    dataAView = dataASqueezed;
                }
                else {
                    std::copy(dataASqueezed.begin(),dataASqueezed.end(),dataCopy.begin());
                    dataAView.assign(sliceShape2.begin(), sliceShape2.end(), &dataCopy(0));
                }

                auto filterA = filterAStorage.getView(0);
                // TODO this needs to be parallelized !
                applyFilters(dataAView, filterA, threadpool);

            }

            // accumulate filter for the in between edges
            if(sliceId < numberOfSlices - 1 && !keepXYOnly) {

                beginB = Coord({int64_t(sliceId+1),int64_t(0),int64_t(0)});
                endB = Coord({int64_t(sliceId+2),shape[1],shape[2]});
                
                auto labelsB = labelsBStorage.getView(0);  
                labelsProxy.readSubarray(beginB, endB, labelsB);
                auto labelsBSqueezed = labelsB.squeezedView();
        
                auto dataB = dataBStorage.getView(0);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = dataB.squeezedView();

                marray::View<float> dataBView;
                if( typeid(DataType) == typeid(float) ) {
                    dataBView = dataBSqueezed;
                }
                else {
                    std::copy(dataBSqueezed.begin(),dataBSqueezed.end(),dataCopy.begin());
                    dataBView.assign(sliceShape2.begin(), sliceShape2.end(), &dataCopy(0));
                }

                auto filterB = filterBStorage.getView(0);
                // TODO this needs to be parallelized !
                applyFilters(dataBView, filterB, threadpool);
            
                // resize the channel acc chain thread vector
                // FIXME why do we use pointers / allocate dynamically here ?
                // I don't know if this makes sense in the setting we have here, discuss with him!
                parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
                [&](const int tid, const int64_t i){
                    perThreadChannelAccChainVector[i] = new ChannelAccChainVectorType( rag.numberOfInBetweenSliceEdges(sliceId),
                        AccChainVectorType( numberOfChannels) );
                    // set minmax
                    auto & perThreadAccChainVector = *(perThreadChannelAccChainVector[i]);
                    for(int64_t edge = 0; edge < rag.numberOfInBetweenSliceEdges(sliceId); ++edge){
                        for(int c = 0; c < numberOfChannels; ++c)
                            perThreadAccChainVector[edge][c].setHistogramOptions(histoOptionsVec[c]);
                    }
                });
                auto betweenEdgeOffset =rag.betweenSliceEdgeOffset(sliceId);
                auto accOffset = keepZOnly ? rag.betweenSliceEdgeOffset(sliceId) - rag.numberOfInSliceEdges() : betweenEdgeOffset;

                // accumulate filter for the between slice edges
                nifty::tools::parallelForEachCoordinate(threadpool, sliceShape2, [&](const int tid, const Coord2 coord){

                    auto & channelAccChainVec = *(perThreadChannelAccChainVector[tid]);
                    // labels are different for different slices by default!
                    const auto lU = labelsASqueezed(coord.asStdArray());
                    const auto lV = labelsBSqueezed(coord.asStdArray());
                    
                    // FIXME I really don't get this vigra coord buisness -> ask thorsten...
                    // do we need 0 or the slice coordinate as 0 entry ???
                    VigraCoord vigraCoordU;    
                    VigraCoord vigraCoordV;    
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = sliceId+1;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord[d-1];
                    }
                    
                    const auto edge = rag.findEdge(lU,lV) - betweenEdgeOffset;
                    
                    for(int c = 0; c < numberOfChannels; ++c) {
                        const auto fU = filterA(c, coord[0], coord[1]);
                        const auto fV = filterB(c, coord[0], coord[1]);
                        channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                        channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
                    }
                });

                // merge
                parallel::parallel_foreach(threadpool, rag.numberOfInBetweenSliceEdges(sliceId), 
                [&](const int tid, const int64_t edge){
                    auto & accChainVec = *(perThreadChannelAccChainVector[0]);
                    for(auto t=1; t<actualNumberOfThreads; ++t){
                        auto & perThreadAccChainVec = *(perThreadChannelAccChainVector[t]);
                        for(int c = 0; c < numberOfChannels; ++c)
                            accChainVec[edge][c].merge(perThreadAccChainVec[edge][c]);
                    }
                });

                f(*perThreadChannelAccChainVector[0], accOffset);
            
                // delete
                parallel::parallel_foreach(threadpool, actualNumberOfThreads, 
                [&](const int tid, const int64_t i){
                    delete perThreadChannelAccChainVector[i];
                });

                // swap A and B
                labelsASqueezed = labelsBSqueezed;
                dataASqueezed = dataBSqueezed;
                filterA = filterB;
            }
        }

    }
    std::cout << "Slices done" << std::endl;
}


// TODO write to hdf5 directly!
// 9 features per channel
template<class LABELS_PROXY, class DATA, class OUTPUT>
void accumulateEdgeFeaturesFromFilters(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    OUTPUT & edgeFeaturesOut,
    const bool keepXYOnly,
    const bool keepZOnly,
    const int numberOfThreads = -1
){
    namespace acc = vigra::acc;
    typedef float DataType;

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

    const auto nStats = 9;

    accumulateEdgeFeaturesFromFiltersWithAccChain<AccChainType>(
        rag,
        data,
        keepXYOnly,
        keepZOnly,
        pOpts,
        threadpool,
        [&](
            const std::vector<std::vector<AccChainType>> & channelAccChainVec,
            const uint64_t edgeOffset
        ){
            using namespace vigra::acc;
            typedef array::StaticArray<int64_t, 2> FeatCoord;

            const auto nEdges    = channelAccChainVec.size();
            const auto nChannels = channelAccChainVec.front().size();

            marray::Marray<DataType> featuresTemp({nEdges,nChannels*nStats});
            
            parallel::parallel_foreach(threadpool, channelAccChainVec.size(),
                [&](const int tid, const int64_t edge){

                const auto & edgeAccChainVec = channelAccChainVec[edge];
                auto cOffset = 0;
                vigra::TinyVector<float,7> quantiles;
                for(int c = 0; c < nChannels; ++c) {
                    const auto & chain = edgeAccChainVec[c];
                    const auto mean = get<acc::Mean>(chain);
                    featuresTemp(edge, cOffset) = replaceIfNotFinite(mean,0.0);
                    featuresTemp(edge, cOffset+1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    quantiles = get<Quantiles>(chain);
                    for(auto qi=0; qi<7; ++qi)
                        featuresTemp(edge, cOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    cOffset += nStats;
                }
            }); 


            FeatCoord begin({int64_t(edgeOffset),0L});
            FeatCoord end({edgeOffset+nEdges,nChannels*nStats});

            //std::cout << "off " << begin << std::endl;
            //std::cout << "tempShape " << featuresTemp.shape(0) << " " << featuresTemp.shape(1) << std::endl;
            //std::cout << "outShape " << edgeFeaturesOut.shape(0) << " " << edgeFeaturesOut.shape(1) << std::endl;

            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        }
    );

}




} // namespace graph
} // namespace nifty

#endif
