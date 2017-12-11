#pragma once

#include <vector>
#include <functional>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "vigra/accumulator.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/tools/array_tools.hxx"

#include "nifty/xtensor/xtensor.hxx"
#include "xtensor/xeval.hpp"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

#ifdef WITH_Z5
#include "nifty/z5/z5.hxx"
#endif

namespace nifty{
namespace graph{

//
// helper functions for accumulate Edge Features
//

// calculate filters for given input with threadpool
template<class DATA_ARRAY, class F, class COORD, class FEATURE_ARRAY>
inline void calculateFilters(const xt::xexpression<DATA_ARRAY> & dataExp,
                             const COORD & sliceShape2,
                             xt::xexpression<FEATURE_ARRAY> & filterExp,
                             parallel::ThreadPool & threadpool,
                             const F & f) {

    typedef typename DATA_ARRAY::value_type DataType;
    typedef COORD Coord;

    const auto & data = dataExp.derived_cast();
    auto & filter = filterExp.derived_cast();

    if( typeid(DataType) == typeid(float) ) {
        std::cout << "Calculating filters mulithreaded w/o copy ..." << std::endl;
        f(data, filter, threadpool);
        std::cout << "done" << std::endl;
    }
    else {
        // copy the data
        // FIXME FIXME FIXME there is probably a faster way yo do this
        typedef typename xt::xtensor<float, 3>::shape_type ShapeType;
        ShapeType shape;
        std::copy(data.shape().begin(), data.shape().end(), shape.begin());
        xt::xtensor<float, 3> dataTmp(shape);
        tools::forEachCoordinate(sliceShape2, [&dataTmp, &data](Coord coord){
            xtensor::write(dataTmp, coord.asStdArray(),
                           (float) xtensor::read(data, coord.asStdArray()));
        });
        std::cout << "Calculating filters mulithreaded w/ copy ..." << std::endl;
        f(dataTmp, filter, threadpool);
        std::cout << "done" << std::endl;
    }
}


// calculate filters for given input single threaded
template<class DATA_ARRAY, class F, class COORD, class FEATURE_ARRAY>
inline void calculateFilters(const xt::xexpression<DATA_ARRAY> & dataExp,
                             const COORD & sliceShape2,
                             xt::xexpression<FEATURE_ARRAY> & filterExp,
                             const F & f,
                             const bool preSmooth = false) {

    typedef typename DATA_ARRAY::value_type DataType;
    typedef COORD Coord;

    const auto & data = dataExp.derived_cast();
    auto & filter = filterExp.derived_cast();

    if( typeid(DataType) == typeid(float) ) {
        std::cout << "Calculating filters singlethreaded w/ copy ..." << std::endl;
        f(data, filter, preSmooth);
        std::cout << "done" << std::endl;
    }
    else {
        // copy the data
        // FIXME FIXME FIXME there is probably a faster way yo do this
        typedef typename xt::xtensor<float, 3>::shape_type ShapeType;
        ShapeType shape;
        std::copy(data.shape().begin(), data.shape().end(), shape.begin());
        xt::xtensor<float, 3> dataTmp(shape);
        tools::forEachCoordinate(sliceShape2, [&data, &dataTmp](Coord coord){
            xtensor::write(dataTmp, coord.asStdArray(),
                           (float) xtensor::read(data, coord.asStdArray()));
        });
        std::cout << "Calculating filters singlethreaded w/o copy ..." << std::endl;
        f(dataTmp, filter, preSmooth);
        std::cout << "done" << std::endl;
    }
}


template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_ARRAY, class RAG, class FEATURE_ARRAY>
inline void accumulateInnerSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
                                         const HISTO_OPTS_VEC & histoOptionsVec,
                                         const COORD & sliceShape2,
                                         const xt::xexpression<LABEL_ARRAY> & labelsExp,
                                         const int64_t sliceId,
                                         const int64_t inEdgeOffset,
                                         const RAG & rag,
                                         const xt::xexpression<FEATURE_ARRAY> & filterExp) {
    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    typedef typename LABEL_ARRAY::value_type LabelType;

    const auto & labels = labelsExp.derived_cast();
    const auto & filter = filterExp.derived_cast();

    size_t pass = 1;
    size_t numberOfChannels = channelAccChainVec[0].size();

    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }

    const auto haveIgnoreLabel = rag.haveIgnoreLabel();
    const auto ignoreLabel = rag.ignoreLabel();

    // accumulate filter for the inner slice edges
    LabelType lU, lV;
    float fU, fV;
    VigraCoord vigraCoordU, vigraCoordV;
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){

        lU = xtensor::read(labels, coord.asStdArray());
        // skip if we have the ignore label activated and
        // if we hit an ignore label
        if(haveIgnoreLabel && lU == ignoreLabel) {
            return;
        }

        for(int axis = 0; axis < 2; ++axis){
            Coord2 coord2 = coord;
            ++coord2[axis];
            if( coord2[axis] < sliceShape2[axis]) {

                lV = xtensor::read(labels, coord2.asStdArray());
                // skip if we have the ignore label activated and
                // if we hit an ignore label
                if(haveIgnoreLabel && lV == ignoreLabel) {
                    return;
                }

                if(lU != lV) {
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = sliceId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord2[d-1];
                    }
                    const auto edge = rag.findEdge(lU,lV) - inEdgeOffset;
                    for(int c = 0; c < numberOfChannels; ++c) {
                        fU = filter(c, coord[0], coord[1]);
                        fV = filter(c, coord2[0], coord2[1]);
                        channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                        channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
                    }
                }
            }
        }
    });
}


// FIXME FIXME FIXME 
// FIXME !!! we waste a lot for zDirection != 0, because we always load both slices, which is totally
// unncessary. Instead, we should have 2 seperate functons (z = 0 / z = 1,2) that get called with the proper
// accumulate filter for the between slice edges
template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_ARRAY, class RAG, class FEATURE_ARRAY>
inline void accumulateBetweenSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
                                           const HISTO_OPTS_VEC & histoOptionsVec,
                                           const COORD & sliceShape2,
                                           const xt::xexpression<LABEL_ARRAY> & labelsAExp,
                                           const xt::xexpression<LABEL_ARRAY> & labelsBExp,
                                           const int64_t sliceIdA,
                                           const int64_t sliceIdB,
                                           const int64_t betweenEdgeOffset,
                                           const RAG & rag,
                                           const xt::xexpression<FEATURE_ARRAY> & filterAExp,
                                           const xt::xexpression<FEATURE_ARRAY> & filterBExp,
                                           const int zDirection){
    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    typedef typename LABEL_ARRAY::value_type LabelType;

    const auto & labelsA = labelsAExp.derived_cast();
    const auto & labelsB = labelsBExp.derived_cast();

    const auto & filterA = filterAExp.derived_cast();
    const auto & filterB = filterBExp.derived_cast();

    const auto haveIgnoreLabel = rag.haveIgnoreLabel();
    const auto ignoreLabel = rag.ignoreLabel();

    size_t pass = 1;
    size_t numberOfChannels = channelAccChainVec[0].size();

    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }

    LabelType lU, lV;
    float fU, fV;
    VigraCoord vigraCoordU, vigraCoordV;
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        // labels are different for different slices by default!
        lU = xtensor::read(labelsA, coord.asStdArray());
        lV = xtensor::read(labelsB, coord.asStdArray());

        // skip if we have the ignore label activated and
        // if we hit an ignore label
        if(haveIgnoreLabel) {
            if(lU == ignoreLabel || lV == ignoreLabel) {
                return;
            }
        }

        vigraCoordU[0] = sliceIdA;
        vigraCoordV[0] = sliceIdB;
        for(int d = 1; d < 3; ++d){
            vigraCoordU[d] = coord[d-1];
            vigraCoordV[d] = coord[d-1];
        }
        const auto edge = rag.findEdge(lU,lV) - betweenEdgeOffset;
        if(zDirection==0) { // 0 -> take into account z and z + 1
            for(int c = 0; c < numberOfChannels; ++c) {
                fU = filterA(c, coord[0], coord[1]);
                fV = filterB(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
            }
        }
        else if(zDirection==1) { // 1 -> take into accout only z
            for(int c = 0; c < numberOfChannels; ++c) {
                fU = filterA(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
            }
        }
        else if(zDirection==2) { // 2 -> take into accout only z + 1
            for(int c = 0; c < numberOfChannels; ++c) {
                fV = filterB(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
            }
        }
    });
}


template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F_XY, class F_Z>
void accumulateEdgeFeaturesFromFiltersWithAccChain(const GridRagStacked2D<LABELS_PROXY> & rag,
                                                   const DATA & data,
                                                   const bool keepXYOnly,
                                                   const bool keepZOnly,
                                                   const parallel::ParallelOptions & pOpts,
                                                   parallel::ThreadPool & threadpool,
                                                   F_XY && fXY,
                                                   F_Z && fZ,
                                                   const int zDirection){
    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::LabelType LabelType;
    typedef typename DATA::value_type DataType;

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
    FiltersToSigmasType filtersToSigmas{std::vector<bool>{true, true, true},  // GaussianSmoothing
                                        std::vector<bool>{true, true, true},  // LaplacianOfGaussian
                                        std::vector<bool>{false, false, false}, // GaussianGradientMagnitude
                                        std::vector<bool>{true,  true,  true}}; // HessianOfGaussianEigenvalues

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

        // TODO we could do this less memory consuming, if we allocate less data here and
        // allocate some in every thread
        LabelBlockStorage  labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage  labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        FilterBlockStorage filterAStorage(threadpool, filterShape, actualNumberOfThreads);
        FilterBlockStorage filterBStorage(threadpool, filterShape, actualNumberOfThreads);
        // we only need one data storage
        DataBlockStorage   dataStorage(threadpool, sliceShape3, actualNumberOfThreads);
        // storage for the data we have to copy if type of data is not float
        //FilterBlockStorage dataCopyStorage(threadpool, sliceShape2, actualNumberOfThreads);

        // process slice 0 to find min and max for histogram opts
        Coord begin0({0L, 0L, 0L});
        Coord end0(  {1L, shape[1], shape[2]});

        auto data0 = dataStorage.getView(0);
        tools::readSubarray(data, begin0, end0, data0);
        auto data0Squeezed = xtensor::squeezedView(data0);
        auto filter0 = filterAStorage.getView(0);

        // apply filters in parallel
        calculateFilters(data0Squeezed,
                         sliceShape2,
                         filter0,
                         threadpool,
                         applyFilters);

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);
        Coord cShape({1L, sliceShape2[0], sliceShape2[1]});
        parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
            auto & histoOpts = histoOptionsVec[c];
            Coord cBegin({c, 0L, 0L});
            xt::slice_vector slice(filter0);
            xtensor::sliceFromOffset(slice, cBegin, cShape);
            auto channelView = xt::dynamic_view(filter0, slice);
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

            // compute the filters for slice A
            Coord beginA({sliceIdA, 0L, 0L});
            Coord endA({sliceIdA+1, shape[1], shape[2]});

            auto labelsA = labelsAStorage.getView(tid);
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = xtensor::squeezedView(labelsA);

            auto dataA = dataStorage.getView(tid);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = xtensor::squeezedView(dataA);

            auto filterA = filterAStorage.getView(tid);
            calculateFilters(dataASqueezed,
                             sliceShape2,
                             filterA,
                             applyFilters,
                             true); // presmoothing

            // acccumulate the inner slice features
            // only if not keepZOnly and if we have at least one edge in this slice
            // (no edge can happend for defected slices)
            if(rag.numberOfInSliceEdges(sliceIdA) > 0 && !keepZOnly) {
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdA);
                // make new acc chain vector
                ChannelAccChainVectorType channelAccChainVec(
                        rag.numberOfInSliceEdges(sliceIdA),
                        AccChainVectorType(numberOfChannels)
                );
                accumulateInnerSliceFeatures(channelAccChainVec,
                                             histoOptionsVec,
                                             sliceShape2,
                                             labelsASqueezed,
                                             sliceIdA,
                                             inEdgeOffset,
                                             rag,
                                             filterA);
                fXY(channelAccChainVec, sliceIdA, inEdgeOffset);
            }

            // read labels, data and calculate the filters for upper slice
            // do if we are not keeping only xy edges or
            // if we are at the last slice (which is never a lower slice and
            // must hence be accumulated extra)
            if(!keepXYOnly || sliceIdB == numberOfSlices - 1) {

                // process upper slice
                Coord beginB = Coord({sliceIdB, 0L, 0L});
                Coord endB   = Coord({sliceIdB + 1, shape[1], shape[2]});
                auto filterB = filterBStorage.getView(tid);

                // read labels
                auto labelsB = labelsBStorage.getView(tid);
                labelsProxy.readSubarray(beginB, endB, labelsB);
                auto labelsBSqueezed = xtensor::squeezedView(labelsB);
                // read data
                auto dataB = dataStorage.getView(tid);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = xtensor::squeezedView(dataB);
                // calc filter
                calculateFilters(dataBSqueezed,
                                 sliceShape2,
                                 filterB,
                                 applyFilters,
                                 true); // activate pre-smoothing

                // acccumulate the between slice features
                if(!keepXYOnly) {
                    auto betweenEdgeOffset = rag.betweenSliceEdgeOffset(sliceIdA);
                    auto accOffset = rag.betweenSliceEdgeOffset(sliceIdA) - rag.numberOfInSliceEdges();
                    // make new acc chain vector
                    ChannelAccChainVectorType channelAccChainVec(
                        rag.numberOfInBetweenSliceEdges(sliceIdA),
                        AccChainVectorType(numberOfChannels)
                    );
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
                                                   filterB,
                                                   zDirection);
                    fZ(channelAccChainVec, sliceIdA, accOffset);
                }

                // accumulate the inner slice features for the last slice, which is never a lower slice
                if(!keepZOnly && (sliceIdB == numberOfSlices - 1 && rag.numberOfInSliceEdges(sliceIdB) > 0)) {
                    auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdB);
                    // make new acc chain vector
                    ChannelAccChainVectorType channelAccChainVec(
                        rag.numberOfInSliceEdges(sliceIdB),
                        AccChainVectorType(numberOfChannels)
                    );
                    accumulateInnerSliceFeatures(channelAccChainVec,
                                                 histoOptionsVec,
                                                 sliceShape2,
                                                 labelsBSqueezed,
                                                 sliceIdB,
                                                 inEdgeOffset,
                                                 rag,
                                                 filterB);
                    fXY(channelAccChainVec, sliceIdB, inEdgeOffset);
                }
            }
        });
    }
    std::cout << "Slices done" << std::endl;
}


template<class OUTPUT, class OVERHANG_STORAGE>
void writeOverhangingChunks(const OVERHANG_STORAGE & overhangsFront,
                            const OVERHANG_STORAGE & overhangsBack,
                            OUTPUT & output,
                            parallel::ThreadPool & threadpool) {
    const auto nSlices = overhangsFront.size();
    NIFTY_CHECK_OP(nSlices, ==, overhangsBack.size(), "len of overhangs must agree");

    // assemble and write the missing chunks in parallel
    parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int64_t sliceId) {

        // we skip id 0
        if(sliceId == 0) {
            return;
        }

        const auto & storageFront = overhangsFront[sliceId];
        const auto & storageBack = overhangsBack[sliceId - 1];

        // make sure that both storages agree whether they have data
        NIFTY_CHECK_OP(storageFront.hasData, ==, storageBack.hasData, "both storages need to have or have not data");
        // if both do not have data, continue
        if(!storageFront.hasData) {
            return;
        }

        const auto & beginCoord = storageFront.begin;
        const auto & endCoord = storageBack.end;

        const auto & dataFront = storageFront.features;
        const auto & dataBack = storageBack.features;

        // assemble the data
        auto dataExp = xt::concatenate(xt::xtuple(dataFront, dataBack), 0);
        auto data = xt::eval(dataExp);

        // write the chunk
        tools::writeSubarray(output, beginCoord, endCoord, data);
    });
}


// 9 features per channel
template<class LABELS_PROXY, class DATA, class OUTPUT>
void accumulateEdgeFeaturesFromFilters(const GridRagStacked2D<LABELS_PROXY> & rag,
                                       const DATA & data,
                                       OUTPUT & edgeFeaturesOutXY,
                                       OUTPUT & edgeFeaturesOutZ,
                                       const bool keepXYOnly,
                                       const bool keepZOnly,
                                       const int zDirection = 0,
                                       const int numberOfThreads = -1) {
    namespace acc = vigra::acc;
    typedef float DataType;

    typedef acc::UserRangeHistogram<40> SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, DataType, SelectType> AccChainType;
    typedef array::StaticArray<int64_t, 2> FeatCoord;

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);

    // use chunked or non-chunkeda accumulation
    if(tools::isChunked(edgeFeaturesOutXY)) {

        const auto & shape = rag.shape();

        // the data for tmp storage of overhanging blocks
        struct OverhangData {
            xt::xtensor<DataType, 2> features;
            FeatCoord begin; // global coordinate where the features begin
            FeatCoord end;   // global coordinate where the features end
            bool hasData;
        };

        // tmp storage for overhanging blocks
        typedef std::vector<OverhangData> OverhangStorage;
        // storages for overhangs front and back
        // NOTE: we won't have overlapping writes here, so having just
        // one big vector is enough
        OverhangStorage storageXYFront(shape[0]);
        OverhangStorage storageXYBack(shape[0]);

        OverhangStorage storageZFront(shape[0]);
        OverhangStorage storageZBack(shape[0]);

        // general accumulator function
        // chunking aware
        auto accFunction = [](
            const std::vector<std::vector<AccChainType>> & channelAccChainVec,
            const size_t sliceId,
            const size_t edgeOffset,
            OUTPUT & edgeFeaturesOut,
            OverhangStorage & storageFront,
            OverhangStorage & storageBack
        ){
            using namespace vigra::acc;
            //typedef array::StaticArray<int64_t, 2> FeatCoord;

            const auto edgeChunkSize = tools::getChunkShape(edgeFeaturesOut)[0];

            const auto nEdges = channelAccChainVec.size();
            const auto nStats = 9;
            const auto nChannels = channelAccChainVec.front().size();
            const auto nFeats = nStats * nChannels;

            xt::xtensor<DataType, 2> featuresTemp({nEdges, nChannels * nStats});
            for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge) {
                const auto & edgeAccChainVec = channelAccChainVec[edge];
                auto cOffset = 0;
                vigra::TinyVector<float, 7> quantiles;
                for(int c = 0; c < nChannels; ++c) {
                    const auto & chain = edgeAccChainVec[c];
                    const auto mean = get<acc::Mean>(chain);
                    featuresTemp(edge, cOffset) = replaceIfNotFinite(mean,0.0);
                    featuresTemp(edge, cOffset+1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    quantiles = get<Quantiles>(chain);
                    for(auto qi=0; qi<7; ++qi) {
                        featuresTemp(edge, cOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                    cOffset += nStats;
                }
            }

            const int64_t edgeEnd = edgeOffset + nEdges;
            const int64_t overhangBegin = (edgeOffset % edgeChunkSize == 0) ? 0 : edgeChunkSize - (edgeOffset % edgeChunkSize);
            const int64_t overhangEnd = edgeEnd % edgeChunkSize;
            // find beginning and end for block-aligned edges in tmp features
            // at the begin, we need to check
            const int64_t edgeEndAlignedLocal = nEdges - overhangEnd;
            FeatCoord beginAlignedLocal{(int64_t)overhangBegin, 0L};
            FeatCoord endAlignedLocal{(int64_t)edgeEndAlignedLocal, (int64_t)nFeats};

            // get view to the aligned features
            xt::slice_vector sliceAligned(featuresTemp);
            xtensor::sliceFromRoi(sliceAligned, beginAlignedLocal, endAlignedLocal);
            auto featuresAligned = xt::dynamic_view(featuresTemp, sliceAligned);

            // find global beginning and end for block aligned edges
            FeatCoord beginAlignedGlobal{int64_t(edgeOffset + overhangBegin), 0L};
            FeatCoord endAlignedGlobal{int64_t(edgeEnd - overhangEnd), (int64_t)nFeats};

            // write out blockaligned edges
            tools::writeSubarray(edgeFeaturesOut, beginAlignedGlobal, endAlignedGlobal, featuresAligned);

            // store non-blockaligned edges (if existing) locally for postprocessing
            // check if we have overhanging data at the beginning
            if(overhangBegin > 0) {
                auto & storage = storageFront[sliceId];
                storage.begin = FeatCoord{int64_t(edgeOffset), 0L};
                storage.end = FeatCoord{int64_t(edgeOffset + overhangBegin), int64_t(nFeats)};

                // write correct data ti the storage
                auto & storageFeats = storage.features;
                std::array<size_t, 2> storageBegin{0, 0};
                std::array<size_t, 2> storageShape{(size_t)overhangBegin, (size_t)nFeats};
                storageFeats.reshape(storageShape);

                xt::slice_vector slice(featuresTemp);
                xtensor::sliceFromOffset(slice, storageBegin, storageShape);
                const auto overhangView = xt::dynamic_view(featuresTemp, slice);
                storageFeats = overhangView;

                storage.hasData = true;
            } else {
                storageFront[sliceId].hasData = false;
            }

            // check if we have overhanging data at the end
            if(overhangEnd > 0) {
                auto & storage = storageBack[sliceId];
                storage.begin = FeatCoord{int64_t(edgeEnd - overhangEnd), 0L};
                storage.end = FeatCoord{int64_t(edgeEnd), int64_t(nFeats)};

                // write correct data ti the storage
                auto & storageFeats = storage.features;
                std::array<size_t, 2> storageBegin{(size_t)edgeEndAlignedLocal, 0};
                std::array<size_t, 2> storageShape{(size_t)overhangEnd, (size_t)nFeats};
                storageFeats.reshape(storageShape);

                xt::slice_vector slice(featuresTemp);
                xtensor::sliceFromOffset(slice, storageBegin, storageShape);
                const auto overhangView = xt::dynamic_view(featuresTemp, slice);
                storageFeats = overhangView;

                storage.hasData = true;
            } else {
                storageBack[sliceId].hasData = false;
            }
        };

        // instantiation of accumulators for xy / z edges
        auto accFunctionXY = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::ref(edgeFeaturesOutXY),
                std::ref(storageXYFront),
                std::ref(storageXYBack));
        auto accFunctionZ = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::ref(edgeFeaturesOutZ),
                std::ref(storageZFront),
                std::ref(storageZBack));

        accumulateEdgeFeaturesFromFiltersWithAccChain<AccChainType>(rag,
                                                                    data,
                                                                    keepXYOnly,
                                                                    keepZOnly,
                                                                    pOpts,
                                                                    threadpool,
                                                                    accFunctionXY,
                                                                    accFunctionZ,
                                                                    zDirection);

        // write the overhanging chunks to file
        // we only need to do this if we actually compute this feature type
        if(!keepZOnly) {
            writeOverhangingChunks(storageXYFront, storageXYBack, edgeFeaturesOutXY, threadpool);
        }

        if(!keepXYOnly) {
            writeOverhangingChunks(storageZFront, storageZBack, edgeFeaturesOutZ, threadpool);
        }

    } else {

        // general accumulator function
        // not chunking aware
        auto accFunction = [](
            const std::vector<std::vector<AccChainType>> & channelAccChainVec,
            const size_t slideId,
            const size_t edgeOffset,
            OUTPUT & edgeFeaturesOut
        ){
            using namespace vigra::acc;

            const auto nStats = 9;
            const auto nEdges = channelAccChainVec.size();
            const auto nChannels = channelAccChainVec.front().size();

            xt::xtensor<DataType, 2> featuresTemp({nEdges, nChannels * nStats});
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
                    for(auto qi=0; qi<7; ++qi) {
                        featuresTemp(edge, cOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    }
                    cOffset += nStats;
                }
            }

            // find beginning and end for block-aligned edges
            FeatCoord begin({(int64_t) edgeOffset, 0L});
            FeatCoord end({int64_t(edgeOffset + nEdges), int64_t(nChannels * nStats)});

            // write out  edges
            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        };

        // instantiation of accumulators for xy / z edges
        auto accFunctionXY = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::ref(edgeFeaturesOutXY));
        auto accFunctionZ = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::ref(edgeFeaturesOutZ));

        accumulateEdgeFeaturesFromFiltersWithAccChain<AccChainType>(rag,
                                                                    data,
                                                                    keepXYOnly,
                                                                    keepZOnly,
                                                                    pOpts,
                                                                    threadpool,
                                                                    accFunctionXY,
                                                                    accFunctionZ,
                                                                    zDirection);
    }
}


// TODO re-enable this
/*
// TODO use the proper helper functions here !
template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
void accumulateSkipEdgeFeaturesFromFiltersWithAccChain(const GridRagStacked2D<LABELS_PROXY> & rag,
                                                       const DATA & data,
                                                       const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
                                                       const std::vector<size_t> & skipRanges,
                                                       const std::vector<size_t> & skipStarts,
                                                       const parallel::ParallelOptions & pOpts,
                                                       parallel::ThreadPool & threadpool,
                                                       F && f,
                                                       const int zDirection){
    typedef std::pair<uint64_t,uint64_t> SkipEdgeStorage;

    typedef LABELS_PROXY LabelsProxyType;
    typedef typename DATA::value_type DataType;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<float> FilterBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;     // holds acc chains over the filter channels for each edge
    typedef std::vector<AccChainVectorType> EdgeAccChainVectorType; // holds acc chains over the edges

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

        // determine the number of skip edges starting from each slice
        // and fill the conversion of uv-ids to edge-id
        std::map<SkipEdgeStorage, size_t> skipUvsToIds;
        for(size_t skipId = 0; skipId < skipEdges.size(); ++skipId) {

            // find the source slice of this skip edge and increment the number of edges in the slice
            auto sliceId = skipStarts[skipId];
            ++numberOfSkipEdgesPerSlice[sliceId];

            // add the target slice of this skip edge to the target slices of the source slice
            auto targetSlice = sliceId + skipRanges[skipId];
            auto & thisSkipSlices = skipSlices[sliceId];
            if(std::find(thisSkipSlices.begin(), thisSkipSlices.end(), targetSlice) == thisSkipSlices.end() )
                thisSkipSlices.push_back(targetSlice);

            // fill the uvs to ids map
            skipUvsToIds[skipEdges[skipId]] = skipId;
        }

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);

        size_t skipEdgeOffset = 0;
        int countSlice = 0;
        for(auto sliceId : lowerSlices) {

            std::cout << countSlice++ << " / " << lowerSlices.size() << std::endl;
            std::cout << "Finding features for skip edges from slice " << sliceId << std::endl;

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

            // init the edge acc vectors for all threads wth the number of skip edges in this slice
            std::vector<EdgeAccChainVectorType> perThreadAccChainVector(actualNumberOfThreads);  // vector<vector<vector<AccChain>>>
            for(int t = 0; t < actualNumberOfThreads; ++t) {                                     // thread edge   channel
                auto & accChainVector = perThreadAccChainVector[t];
                accChainVector.resize(numberOfSkipEdgesInSlice);

                // set the histo opts for each edge acc chain vector and channel
                parallel::parallel_foreach(threadpool, numberOfSkipEdgesInSlice, [&](const int tid, const int64_t skipEdge){
                        auto & accChainVec = accChainVector[skipEdge];
                        accChainVec.resize(numberOfChannels);
                        for(int c = 0; c < numberOfChannels; ++c){
                            accChainVec[c].setHistogramOptions(histoOptionsVec[c]);
                        }
                });
            }

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

                    // we search for the skipId corresponding to this pair of uv-ids
                    // for slices with partial defects, not all uv-pairs correspond to a skip-edge.
                    // in that case, we simply continue without doing anything
                    auto skipPair = std::make_pair(static_cast<uint64_t>(lU), static_cast<uint64_t>(lV));
                    auto mapIterator = skipUvsToIds.find(skipPair);
                    if(mapIterator == skipUvsToIds.end()) {
                        return;
                    }

                    // the in slice id corresponds to the global id - the offset corresponding to
                    // the current source slice
                    auto skipId = mapIterator->second - skipEdgeOffset;

                    auto & accChainVec = threadData[skipId];

                    VigraCoord vigraCoordU;
                    VigraCoord vigraCoordV;
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = nextId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord[d-1];
                    }

                    if(zDirection==0) { // 0 -> take into account z and z + 1
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fU = filterA(c, coord[0], coord[1]);
                            const auto fV = filterB(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fU, vigraCoordU, pass);
                            accChainVec[c].updatePassN(fV, vigraCoordV, pass);
                        }
                    }
                    else if(zDirection==1) { // 1 -> take into accout only z
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fU = filterA(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fU, vigraCoordU, pass);
                        }
                    }
                    else if(zDirection==2) { // 2 -> take into accout only z + 1
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fV = filterB(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fV, vigraCoordV, pass);
                        }
                    }
                });
            }

            // we merge the accumulator chains into the zeros acc chain in parallel over the skip edges
            auto & accumulatorVectorDest = perThreadAccChainVector[0];
            parallel::parallel_foreach(threadpool, numberOfSkipEdgesInSlice, [&](const int tid, const int64_t skipId){

                auto & accVecDest = accumulatorVectorDest[skipId];

                for(int t = 1; t < actualNumberOfThreads; ++t) {
                    auto & accVecSrc = perThreadAccChainVector[t][skipId];
                     for(int c = 0; c < numberOfChannels; ++c) {
                        accVecDest[c].merge(accVecSrc[c]);
                    }
                }
            });

            f(accumulatorVectorDest, skipEdgeOffset);
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
    const int zDirection = 0,
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

            xt::xtensor<DataType, 2> featuresTemp({nEdges, nChannels * nStats});

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
        },
        zDirection
    );
}
*/


} // namespace graph
} // namespace nifty
