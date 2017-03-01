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
#include "nifty/tools/array_tools.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

namespace nifty{
namespace graph{

//
// helper functions for accumulate Edge Features
//

// calculate filters for given input with threadpool
template<class DATA_TYPE, class F, class COORD>
inline void calculateFilters(const marray::View<DATA_TYPE> & dataSqueezed,
        marray::View<float> & dataCopy,
        const COORD & sliceShape2,
        marray::View<float> & filter,
        parallel::ThreadPool & threadpool,
        const F & f) {
    
    typedef DATA_TYPE DataType;
    typedef COORD Coord;
    if( typeid(DataType) == typeid(float) ) {
        dataCopy = dataSqueezed;
    }
    else {
        // copy the data (we don't use std::copy here, because iterators are terribly
        // slow for marrays)
        tools::forEachCoordinate(
            sliceShape2, [&dataCopy,&dataSqueezed](Coord coord){
                dataCopy(coord.asStdArray()) = (float) dataSqueezed(coord.asStdArray());        
        });
    }
    f(dataCopy, filter, threadpool);
}

// calculate filters for given input single threaded
// TODO use pre-smoothing once implemented
template<class DATA_TYPE, class F, class COORD>
inline void calculateFilters(const marray::View<DATA_TYPE> & dataSqueezed,
        marray::View<float> & dataCopy,
        const COORD & sliceShape2,
        marray::View<float> & filter,
        const F & f,
        const bool preSmooth = false) {
    
    typedef DATA_TYPE DataType;
    typedef COORD Coord;
    if( typeid(DataType) == typeid(float) ) {
        dataCopy = dataSqueezed;
    }
    else {
        // copy the data (we don't use std::copy here, because iterators are terribly
        // slow for marrays)
        tools::forEachCoordinate(
            sliceShape2, [&dataCopy,&dataSqueezed](Coord coord){
                dataCopy(coord.asStdArray()) = (float) dataSqueezed(coord.asStdArray());        
        });
    }
    f(dataCopy, filter, preSmooth);
}

template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_TYPE, class RAG>
inline void accumulateInnerSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
        const HISTO_OPTS_VEC & histoOptionsVec,
        const COORD & sliceShape2,
        const marray::View<LABEL_TYPE> & labelsSqueezed,
        const int64_t sliceId,
        const int64_t inEdgeOffset,
        const RAG & rag,
        const marray::View<float> & filter
        ) {
    
    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    size_t pass = 1;
    size_t numberOfChannels = channelAccChainVec[0].size();
    
    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }
    
    // accumulate filter for the inner slice edges
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        const auto lU = labelsSqueezed(coord.asStdArray());
        for(int axis = 0; axis < 2; ++axis){
            Coord2 coord2 = coord;
            ++coord2[axis];
            if( coord2[axis] < sliceShape2[axis]) {
                const auto lV = labelsSqueezed(coord2.asStdArray());
                if(lU != lV) {
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
                        const auto fU = filter(c, coord[0], coord[1]);
                        const auto fV = filter(c, coord2[0], coord2[1]);
                        channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                        channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
                    }
                }
            }
        }
    });
}

// accumulate filter for the between slice edges
template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_TYPE, class RAG>
inline void accumulateBetweenSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
        const HISTO_OPTS_VEC & histoOptionsVec,
        const COORD & sliceShape2,
        const marray::View<LABEL_TYPE> & labelsASqueezed,
        const marray::View<LABEL_TYPE> & labelsBSqueezed,
        const int64_t sliceIdA,
        const int64_t sliceIdB,
        const int64_t betweenEdgeOffset,
        const RAG & rag,
        const marray::View<float> & filterA,
        const marray::View<float> & filterB
    ){
    
    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    size_t pass = 1;
    size_t numberOfChannels = channelAccChainVec[0].size();
    
    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }
            
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        // labels are different for different slices by default!
        const auto lU = labelsASqueezed(coord.asStdArray());
        const auto lV = labelsBSqueezed(coord.asStdArray());
        VigraCoord vigraCoordU;    
        VigraCoord vigraCoordV;    
        vigraCoordU[0] = sliceIdA;
        vigraCoordV[0] = sliceIdB;
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
}


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
    typedef typename LabelsProxyType::LabelType LabelType;
    typedef typename DATA::DataType DataType;
    
    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<float> FilterBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;
    
    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType; 
    typedef std::vector<AccChainVectorType> ChannelAccChainVectorType; 
    typedef typename features::ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();
    
    // sigmas and filters to sigmas: TODO make accessible
    std::vector<double> sigmas({1.6,4.2,8.2});
    FiltersToSigmasType filtersToSigmas({ { true, true, true},      // GaussianSmoothing
                                          { true, true, true},      // LaplacianOfGaussian
                                          { false, false, false},   // GaussianGradientMagnitude
                                          { true, true, true } });  // HessianOfGaussianEigenvalues
    
    features::ApplyFilters<2> applyFilters(sigmas, filtersToSigmas);
    size_t numberOfChannels = applyFilters.numberOfChannels();
    
    uint64_t numberOfSlices = shape[0];
    
    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1L, shape[1], shape[2]});
    Coord filterShape({int64_t(numberOfChannels), shape[1], shape[2]});
    
    // filter computation and accumulation
    // FIXME we only support 1 pass for now
    //for(auto pass = 1; pass <= channelAccChainVector.front().front().passesRequired(); ++pass) {
    //int pass = 1;
    {
        // accumulate inner slice feature

        // edge acc vectors for multiple threads
        std::vector< ChannelAccChainVectorType> perThreadChannelAccChainVector(actualNumberOfThreads);

        LabelBlockStorage  labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage  labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        FilterBlockStorage filterAStorage(threadpool, filterShape, actualNumberOfThreads);
        FilterBlockStorage filterBStorage(threadpool, filterShape, actualNumberOfThreads);
        // we only need one data storage
        DataBlockStorage   dataStorage(threadpool, sliceShape3, actualNumberOfThreads);
        // storage for the data we have to copy if type of data is not float
        FilterBlockStorage dataCopyStorage(threadpool, sliceShape2, actualNumberOfThreads);

        // process slice 0 to find min and max for histogram opts
        Coord begin0({0L, 0L, 0L}); 
        Coord end0(  {1L, shape[1], shape[2]}); 
        
        auto labels0 = labelsAStorage.getView(0);  
        labelsProxy.readSubarray(begin0, end0, labels0);
        auto labels0Squeezed = labels0.squeezedView();
            
        auto data0 = dataStorage.getView(0);
        tools::readSubarray(data, begin0, end0, data0);
        auto data0Squeezed = data0.squeezedView();

        auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
        auto filter0 = filterAStorage.getView(0);
        // apply filters in parallel
        calculateFilters(data0Squeezed,
                dataCopy,
                sliceShape2,
                filter0,
                threadpool,
                applyFilters);
        // apply filters with pre smoothing
        // TODO benchmark this!
        //calculateFilters(data0Squeezed,
        //        dataCopy,
        //        sliceShape2,
        //        filter0,
        //        applyFilters,
        //        true);

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);
        Coord cShape({1L,sliceShape2[0],sliceShape2[1]});
        parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
            auto & histoOpts = histoOptionsVec[c];
            Coord cBegin({c,0L,0L});
            auto channelView = filter0.view(cBegin.begin(), cShape.begin());
            auto minMax = std::minmax_element(channelView.begin(), channelView.end());
            auto min = *(minMax.first); 
            auto max = *(minMax.second);
            histoOpts.setMinMax(min,max);
        });

        // construct slice pairs for processing in parallel
        std::vector<std::pair<int64_t,int64_t>> slicePairs;
        int64_t lowerSliceId = 0;
        int64_t upperSliceId = 1;
        while(upperSliceId < numberOfSlices) {
            slicePairs.emplace_back(std::make_pair(lowerSliceId,upperSliceId));
            ++lowerSliceId;
            ++upperSliceId;
        }
    
        parallel::parallel_foreach(threadpool, slicePairs.size(), [&](const int tid, const int64_t pairId){

            std::cout << "Processing slice pair: " << pairId << " / " << slicePairs.size() << std::endl;
            int64_t sliceIdA = slicePairs[pairId].first; // lower slice
            int64_t sliceIdB = slicePairs[pairId].second;// upper slice
            //std::cout << "Upper: " << sliceIdA << " Lower: " << sliceIdB << std::endl;
            auto & channelAccChainVec = perThreadChannelAccChainVector[tid];

            // compute the filters for slice A
            Coord beginA ({sliceIdA, 0L, 0L});
            Coord endA({sliceIdA+1, shape[1], shape[2]});
            
            auto labelsA = labelsAStorage.getView(tid);  
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();
        
            auto dataA = dataStorage.getView(tid);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = dataA.squeezedView();
            auto dataCopy = dataCopyStorage.getView(tid);
            auto filterA = filterAStorage.getView(tid);
            calculateFilters(dataASqueezed,
                    dataCopy,
                    sliceShape2,
                    filterA,
                    applyFilters,
                    true); // presmoothing

            // acccumulate the inner slice features
            // only if not keepZOnly and if we have at least one edge in this slice
            // (no edge can happend for defected slices)
            if( rag.numberOfInSliceEdges(sliceIdA) > 0 && !keepZOnly) {
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdA);
                // resize the current channel acc chain vector
                channelAccChainVec = ChannelAccChainVectorType( rag.numberOfInSliceEdges(sliceIdA),
                        AccChainVectorType(numberOfChannels) );
                accumulateInnerSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsASqueezed,
                        sliceIdA,
                        inEdgeOffset,
                        rag,
                        filterA);
                f(channelAccChainVec, inEdgeOffset);
            }

            // process upper slice
            Coord beginB = Coord({sliceIdB,   0L,       0L});
            Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});
            auto filterB = filterBStorage.getView(tid);
            marray::View<LabelType> labelsBSqueezed;
        
            // read labels, data and calculate the filters for upper slice
            // do if we are not keeping only xy edges or
            // if we are at the last slice (which is never a lower slice and 
            // must hence be accumulated extra)
            if(!keepXYOnly || sliceIdB == numberOfSlices - 1 ) {
                // read labels
                auto labelsB = labelsBStorage.getView(tid);  
                labelsProxy.readSubarray(beginB, endB, labelsB);
                labelsBSqueezed = labelsB.squeezedView();
                // read data
                auto dataB = dataStorage.getView(tid);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = dataB.squeezedView();
                // calc filter
                calculateFilters(dataBSqueezed,
                        dataCopy,
                        sliceShape2,
                        filterB,
                        applyFilters,
                        true); // activate pre-smoothing
            }
            
            // acccumulate the between slice features
            if(!keepXYOnly) {
                auto betweenEdgeOffset = rag.betweenSliceEdgeOffset(sliceIdA);
                auto accOffset = keepZOnly ? rag.betweenSliceEdgeOffset(sliceIdA) - rag.numberOfInSliceEdges() : betweenEdgeOffset;
                // resize the current channel acc chain vector
                channelAccChainVec = ChannelAccChainVectorType( rag.numberOfInBetweenSliceEdges(sliceIdA),
                        AccChainVectorType(numberOfChannels) );
                // accumulate features for the in between slice edges
                accumulateBetweenSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsASqueezed,
                        labelsBSqueezed,
                        sliceIdA,
                        sliceIdB,
                        betweenEdgeOffset,
                        rag,
                        filterA,
                        filterB);
                f(channelAccChainVec, accOffset);
            }
               
            // accumulate the inner slice features for the last slice
            if(!keepZOnly && (sliceIdB == numberOfSlices - 1 && rag.numberOfInSliceEdges(sliceIdB) > 0)) {
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdB);
                // resize the current channel acc chain vector
                channelAccChainVec = ChannelAccChainVectorType( rag.numberOfInSliceEdges(sliceIdB),
                        AccChainVectorType(numberOfChannels) );
                accumulateInnerSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsBSqueezed,
                        sliceIdB,
                        inEdgeOffset,
                        rag,
                        filterB);
                f(channelAccChainVec, inEdgeOffset);
            }

        });
    }
    std::cout << "Slices done" << std::endl;
}


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
            for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge) {
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
            } 

            FeatCoord begin({int64_t(edgeOffset),0L});
            FeatCoord end({edgeOffset+nEdges,nChannels*nStats});

            //std::cout << "off " << begin << std::endl;
            //std::cout << "tempShape " << featuresTemp.shape(0) << " " << featuresTemp.shape(1) << std::endl;
            //std::cout << "outShape " << edgeFeaturesOut.shape(0) << " " << edgeFeaturesOut.shape(1) << std::endl;

            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        }
    );
}


// TODO use the proper helper functions here !
template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
void accumulateSkipEdgeFeaturesFromFiltersWithAccChain(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
    const std::vector<size_t> & skipRanges,
    const std::vector<size_t> & skipStarts,
    const parallel::ParallelOptions & pOpts,
    parallel::ThreadPool & threadpool,
    F && f
){
    typedef std::pair<uint64_t,uint64_t> SkipEdgeStorage; 
    
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
    typedef std::map<SkipEdgeStorage,AccChainVectorType> ChannelAccChainMapType; 
    typedef std::vector<AccChainVectorType> ChannelAccChainVectorType; 
    typedef typename features::ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();
    
    // sigmas and filters to sigmas: TODO make accessible
    std::vector<double> sigmas({1.6,4.2,8.2});
    FiltersToSigmasType filtersToSigmas({ { true, true, true},      // GaussianSmoothing
                                          { true, true, true},      // LaplacianOfGaussian
                                          { false, false, false},   // GaussianGradientMagnitude
                                          { true, true, true } });  // HessianOfGaussianEigenvalues
    
    features::ApplyFilters<2> applyFilters(sigmas, filtersToSigmas);
    size_t numberOfChannels = applyFilters.numberOfChannels();
    
    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1L,shape[1], shape[2]});
    Coord filterShape({int64_t(numberOfChannels), shape[1], shape[2]});
    
    // filter computation and accumulation
    // FIXME we only support 1 pass for now
    //for(auto pass = 1; pass <= channelAccChainVector.front().front().passesRequired(); ++pass) {
    int pass = 1;
    {
        LabelBlockStorage labelsAStorage(threadpool, sliceShape3, 1);
        LabelBlockStorage labelsBStorage(threadpool, sliceShape3, 1);
        FilterBlockStorage filterAStorage(threadpool, filterShape, 1);
        FilterBlockStorage filterBStorage(threadpool, filterShape, 1);
        // only need one data storage
        DataBlockStorage dataStorage(threadpool, sliceShape3, 1);
        // storage for the data we need to copy for uint8 input
        FilterBlockStorage dataCopyStorage(threadpool, sliceShape2, 1);

        // get unique lower slices with skip edges
        std::vector<size_t> lowerSlices;
        tools::uniques(skipStarts, lowerSlices);
        auto lowest = int64_t(lowerSlices[0]);

        // get upper slices with skip edges for each lower slice and number of skip edges for each lower slice
        std::map<size_t,std::vector<size_t>> skipSlices;
        std::map<size_t,size_t> numberOfSkipEdgesPerSlice;
        // initialize the maps
        std::cout << "Lower slices with skip edges:" << std::endl;
        for(auto sliceId : lowerSlices) {
            std::cout << sliceId << std::endl;
            skipSlices[sliceId] = std::vector<size_t>();
            numberOfSkipEdgesPerSlice[sliceId] = 0;
        }
        // 
        for(size_t skipId = 0; skipId < skipEdges.size(); ++skipId) {
            auto sliceId = skipStarts[skipId];
            ++numberOfSkipEdgesPerSlice[sliceId];
            auto targetSlice = sliceId + skipRanges[skipId];
            auto & thisSkipSlices = skipSlices[sliceId];
            if(std::find(thisSkipSlices.begin(), thisSkipSlices.end(), targetSlice) == thisSkipSlices.end() )
                thisSkipSlices.push_back(targetSlice);
        }
        
        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);
        
        size_t skipEdgeOffset = 0;
        int countSlice = 0;
        for(auto sliceId : lowerSlices) {

            std::cout << countSlice++ << " / " << lowerSlices.size() << std::endl;
            std::cout << "Finding features for skip edges from slice " << sliceId << std::endl; 
        
            // edge acc vectors for multiple threads
            std::vector< ChannelAccChainMapType> perThreadAccChainVector(actualNumberOfThreads);
                
            Coord beginA({int64_t(sliceId),0L,0L}); 
            Coord endA(  {int64_t(sliceId+1),shape[1],shape[2]}); 
            auto labelsA = labelsAStorage.getView(0);  
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();
        
            auto dataA = dataStorage.getView(0);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = dataA.squeezedView();

            auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
            auto filterA = filterAStorage.getView(0);
            // apply filters in parallel
            calculateFilters(dataASqueezed,
                    dataCopy,
                    sliceShape2,
                    filterA,
                    threadpool,
                    applyFilters);
            
            // set the correct histogram for each filter from lowest slice
            if(sliceId == lowest){
                Coord cShape({1L,sliceShape2[0],sliceShape2[1]});
                parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
                    auto & histoOpts = histoOptionsVec[c];
                    
                    Coord cBegin({c,0L,0L});
                    auto channelView = filterA.view(cBegin.begin(), cShape.begin());
                    auto minMax = std::minmax_element(channelView.begin(), channelView.end());
                    auto min = *(minMax.first);
                    auto max = *(minMax.second); 
                    histoOpts.setMinMax(min,max);
                });
            }
                
            auto numberOfSkipEdgesInSlice = numberOfSkipEdgesPerSlice[sliceId];
            
            Coord beginB;
            Coord endB;
            // iterate over all upper slices that have skip edges with this slice
            for(auto nextId : skipSlices[sliceId] ) {
                std::cout << "to slice " << nextId << std::endl;

                beginB = Coord({int64_t(nextId),0L,0L});
                endB   = Coord({int64_t(nextId+1),shape[1],shape[2]});
                
                auto labelsB = labelsBStorage.getView(0);  
                labelsProxy.readSubarray(beginB, endB, labelsB);
                auto labelsBSqueezed = labelsB.squeezedView();
        
                auto dataB = dataStorage.getView(0);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = dataB.squeezedView();
            
                auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
                auto filterB = filterBStorage.getView(0);
                // apply filters in parallel
                calculateFilters(dataBSqueezed,
                        dataCopy,
                        sliceShape2,
                        filterB,
                        threadpool,
                        applyFilters);
            
                // accumulate filter for the between slice edges
                nifty::tools::parallelForEachCoordinate(threadpool, sliceShape2, [&](const int tid, const Coord2 coord){

                    auto & threadData = perThreadAccChainVector[tid];
                    
                    // labels are different for different slices by default!
                    const auto lU = labelsASqueezed(coord.asStdArray());
                    const auto lV = labelsBSqueezed(coord.asStdArray());

                    auto skipPair = std::make_pair(static_cast<uint64_t>(lU), static_cast<uint64_t>(lV));
                    auto skipIt   = threadData.find(skipPair);

                    if(skipIt == threadData.end()) {
                    
                        // first time we hit that edge -> initialize the vector with AccChaains for the different channels
                        threadData[skipPair] = AccChainVectorType();
                        auto & accChainVec = threadData[skipPair];

                        for(int c = 0; c < numberOfChannels; ++c){
                            accChainVec.emplace_back( EdgeAccChainType() );
                            accChainVec[c].setHistogramOptions(histoOptionsVec[c]);
                        }
                    }
                    
                    auto & accChainVec = threadData[skipPair];
                        
                    VigraCoord vigraCoordU;    
                    VigraCoord vigraCoordV;    
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = nextId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord[d-1];
                    }
                    
                    for(int c = 0; c < numberOfChannels; ++c) {
                        const auto fU = filterA(c, coord[0], coord[1]);
                        const auto fV = filterB(c, coord[0], coord[1]);
                        accChainVec[c].updatePassN(fU, vigraCoordU, pass);
                        accChainVec[c].updatePassN(fV, vigraCoordV, pass);
                    }
                });
            }
            
            // merge
            // init the edge chain vector we merge stuff to
            ChannelAccChainVectorType accChainVector(numberOfSkipEdgesInSlice);
            // TODO init for all in slice skip edges
            parallel::parallel_foreach(threadpool, numberOfSkipEdgesInSlice, 
                [&](const int tid, const int64_t skipEdge){
                    auto & accChainVec = accChainVector[skipEdge];
                    for(int c = 0; c < numberOfChannels; ++c){
                        accChainVec.emplace_back( EdgeAccChainType() );
                        accChainVec[c].setHistogramOptions(histoOptionsVec[c]);
                    }
            });
            
            // merge the maps for actual skip edges
            for(size_t t = 0; t < actualNumberOfThreads; ++t) {
                    
                auto & threadData = perThreadAccChainVector[t];
                // get the keys from the map holding all potential skip edges
                std::vector<SkipEdgeStorage> keys;
                tools::extractKeys(threadData, keys);
                parallel::parallel_foreach(threadpool, keys.size(), 
                [&](const int tid, const int64_t keyId){
                    
                    auto & key = keys[keyId];
                    auto skipIterator = std::find(skipEdges.begin()+skipEdgeOffset, skipEdges.end()+skipEdgeOffset+numberOfSkipEdgesInSlice, key);
                    if(skipIterator != skipEdges.end()) {
                        const auto skipId = std::distance(skipEdges.begin()+skipEdgeOffset, skipIterator);
                        auto & accVecSrc  = threadData[key];
                        auto & accVecDest = accChainVector[skipId];
                        for(int c = 0; c < numberOfChannels; ++c)
                            accVecDest[c].merge(accVecSrc[c]);
                    }
                });

            }

            f(accChainVector, skipEdgeOffset);
            skipEdgeOffset += numberOfSkipEdgesInSlice;
        }
    }
}


// 9 features per channel
template<class LABELS_PROXY, class DATA, class OUTPUT>
void accumulateSkipEdgeFeaturesFromFilters(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    OUTPUT & edgeFeaturesOut,
    const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
    const std::vector<size_t> & skipRanges,
    const std::vector<size_t> & skipStarts,
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

    accumulateSkipEdgeFeaturesFromFiltersWithAccChain<AccChainType>(
        rag,
        data,
        skipEdges,
        skipRanges,
        skipStarts,
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

            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        }
    );
}


} // namespace graph
} // namespace nifty

#endif
