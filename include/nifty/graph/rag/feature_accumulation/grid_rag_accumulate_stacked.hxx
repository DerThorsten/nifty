#pragma once

#include <vector>
#include <cmath>

#include "xtensor/xtensor.hpp"
#include "xtensor/xeval.hpp"

#include "nifty/tools/array_tools.hxx"

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"

#include "nifty/xtensor/xtensor.hxx"

namespace nifty{
namespace graph{

    template<class ACC_CHAIN_VECTOR, class HISTO_OPTS, class COORD, class LABEL_ARRAY, class DATA_ARRAY, class RAG>
    inline void accumulateInnerSliceFeatures(
            ACC_CHAIN_VECTOR & accChainVec,
            const HISTO_OPTS & histoOptions,
            const COORD & sliceShape2,
            const xt::xexpression<LABEL_ARRAY> & labelsExp,
            const int64_t sliceId,
            const int64_t inEdgeOffset,
            const RAG & rag,
            const xt::xexpression<DATA_ARRAY> & dataExp,
            const int pass
        ) {

        typedef COORD Coord2;
        typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
        typedef typename LABEL_ARRAY::value_type LabelType;

        const auto & labels = labelsExp.derived_cast();
        const auto & data = dataExp.derived_cast();

        const auto haveIgnoreLabel = rag.haveIgnoreLabel();
        const auto ignoreLabel = rag.ignoreLabel();

        // set minmax for accumulator chains
        for(int64_t edge = 0; edge < accChainVec.size(); ++edge){
            accChainVec[edge].setHistogramOptions(histoOptions);
        }

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
                if(coord2[axis] < sliceShape2[axis]) {

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
                        fU = xtensor::read(data, coord.asStdArray());
                        fV = xtensor::read(data, coord2.asStdArray());
                        accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                        accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
                    }
                }
            }
        });
    }


    // FIXME FIXME FIXME 
    // FIXME !!! we waste a lot for zDirection != 0, because we always load both slices, which is totally
    // unncessary. Instead, we should have 2 seperate functons (z = 0 / z = 1,2) that get called with the proper 
    // labels and data
    //
    // accumulate filter for the between slice edges
    template<class ACC_CHAIN_VECTOR, class HISTO_OPTS, class COORD, class LABEL_ARRAY, class DATA_ARRAY, class RAG>
    inline void accumulateBetweenSliceFeatures(
            ACC_CHAIN_VECTOR & accChainVec,
            const HISTO_OPTS & histoOptions,
            const COORD & sliceShape2,
            const xt::xexpression<LABEL_ARRAY> & labelsAExp,
            const xt::xexpression<LABEL_ARRAY> & labelsBExp,
            const int64_t sliceIdA,
            const int64_t sliceIdB,
            const int64_t betweenEdgeOffset,
            const RAG & rag,
            const xt::xexpression<DATA_ARRAY> & dataAExp,
            const xt::xexpression<DATA_ARRAY> & dataBExp,
            const int zDirection,
            const int pass
        ){

        typedef COORD Coord2;
        typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
        typedef typename LABEL_ARRAY::value_type LabelType;

        const auto & labelsA = labelsAExp.derived_cast();
        const auto & labelsB = labelsBExp.derived_cast();

        const auto & dataA = dataAExp.derived_cast();
        const auto & dataB = dataBExp.derived_cast();

        const auto haveIgnoreLabel = rag.haveIgnoreLabel();
        const auto ignoreLabel = rag.ignoreLabel();

        // set minmax for accumulator chains
        for(int64_t edge = 0; edge < accChainVec.size(); ++edge){
            accChainVec[edge].setHistogramOptions(histoOptions);
        }

        VigraCoord vigraCoordU, vigraCoordV;
        LabelType lU, lV;
        float fU, fV;
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
                fU = xtensor::read(dataA, coord.asStdArray());
                fV = xtensor::read(dataB, coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
            else if(zDirection==1) { // 1 -> take into accout only z
                fU = xtensor::read(dataA, coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
            }
            else if(zDirection==2) { // 2 -> take into accout only z + 1
                fV = xtensor::read(dataB, coord.asStdArray());
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
        });
    }


    // FIXME FIXME FIXME
    // I am pretty sure this does not do what I inteded
    // (i.e. it copys float as well as every other data type)
    // copy data if the dtype is not float
    template<class ARRAY1, class ARRAY2, class COORD>
    inline void copyIfNecessary(const xt::xexpression<ARRAY1> & dataSqueezedExp,
                                xt::xexpression<ARRAY2> & dataCopyExp,
                                const COORD & sliceShape2) {
        typedef typename ARRAY1::value_type DataType;
        typedef COORD Coord;
        auto & dataSqueezed = dataSqueezedExp.derived_cast();
        auto & dataCopy = dataCopyExp.derived_cast();
        if( typeid(DataType) == typeid(float) ) {
            dataCopy = dataSqueezed;
            return;
        }
        else {
            tools::forEachCoordinate(
                sliceShape2, [&dataCopy, &dataSqueezed](Coord coord){
                    xtensor::write(dataCopy, coord.asStdArray(),
                                   (float) xtensor::read(dataSqueezed, coord.asStdArray()));
            });
            return;
        }
    }


    template<class OUTPUT, class OVERHANG_STORAGE>
    void writeOverhangingChunks(const OVERHANG_STORAGE & overhangsFront,
                                const OVERHANG_STORAGE & overhangsBack,
                                OUTPUT & output,
                                parallel::ThreadPool & threadpool,
                                const bool forZEdges) {
        const auto nSlices = overhangsFront.size();
        NIFTY_CHECK_OP(nSlices, ==, overhangsBack.size(), "len of overhangs must agree");

        // assemble and write the missing chunks in parallel
        parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int64_t sliceId) {

            // we skip slice 0, because there are no overhangs to assemble
            if(sliceId == 0) {
                return;
            }

            const auto & storageFront = overhangsFront[sliceId];
            const auto & storageBack = overhangsBack[sliceId - 1];

            // need to write last overhangs for z-edges
            if(sliceId == nSlices - 1 && forZEdges) {
                const auto & beginCoord = storageBack.begin;
                const auto & endCoord = storageBack.end;
                const auto & dataBack = storageBack.features;
                tools::writeSubarray(output, beginCoord, endCoord, dataBack);
                return;
            }

            // make sure that both storages agree whether they have data
            NIFTY_CHECK_OP(storageFront.hasData, ==, storageBack.hasData, "both storages need to have or have not data");
            // if both do not have data, continue
            if(!storageFront.hasData) {
                return;
            }

            const auto & beginCoord = storageBack.begin;
            const auto & endCoord = storageFront.end;

            const auto & dataFront = storageFront.features;
            const auto & dataBack = storageBack.features;

            // assemble the data
            auto dataExp = xt::concatenate(xt::xtuple(dataBack, dataFront), 0);
            auto data = xt::eval(dataExp);

            // write the chunk
            tools::writeSubarray(output, beginCoord, endCoord, data);
        });

        // need to write the last overhangs for xy edges
        if(!forZEdges) {
            const auto & storageBack = overhangsBack[nSlices-1];
            const auto & beginCoord = storageBack.begin;
            const auto & endCoord = storageBack.end;
            const auto & dataBack = storageBack.features;
            tools::writeSubarray(output, beginCoord, endCoord, dataBack);
        }
    }


    // accumulator with data
    template<class EDGE_ACC_CHAIN, class LABELS, class DATA, class F_XY, class F_Z>
    void accumulateEdgeFeaturesWithAccChain(
        const GridRagStacked2D<LABELS> & rag,
        const DATA & data,
        const bool keepXYOnly,
        const bool keepZOnly,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F_XY && fXY,
        F_Z && fZ,
        const int zDirection
    ){

        typedef LABELS LabelsType;
        typedef typename LabelsType::value_type LabelType;
        typedef typename DATA::value_type DataType;

        typedef typename vigra::MultiArrayShape<3>::type   VigraCoord;
        typedef typename GridRagStacked2D<LabelsType>::BlockStorageType LabelStorage;
        typedef tools::BlockStorage<DataType> DataStorage;
        typedef tools::BlockStorage<float> DataCopyStorage;

        typedef array::StaticArray<int64_t,3> Coord;
        typedef array::StaticArray<int64_t,2> Coord2;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType;

        const auto & labels = rag.labels();
        const auto & shape = rag.shape();

        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        const Coord2 sliceShape2({shape[1], shape[2]});
        const Coord  sliceShape3({1L,shape[1], shape[2]});

        // For now, we only support single pass!
        // do N passes of accumulator
        //for(auto pass=1; pass <= passesRequired; ++pass){
        int pass = 1;
        {
            LabelStorage labelsAStorage(threadpool, sliceShape3, nThreads);
            LabelStorage labelsBStorage(threadpool, sliceShape3, nThreads);
            DataStorage dataAStorage(threadpool, sliceShape3, nThreads);
            DataStorage dataBStorage(threadpool, sliceShape3, nThreads);

            // storage for the data we have to copy if type of data is not float
            DataCopyStorage dataACopyStorage(threadpool, sliceShape2, nThreads);
            DataCopyStorage dataBCopyStorage(threadpool, sliceShape2, nThreads);

            // process slice 0 to find min and max for histogram opts
            Coord begin0({0L, 0L, 0L});
            Coord end0(  {1L, shape[1], shape[2]});

            auto data0 = dataAStorage.getView(0);
            tools::readSubarray(data, begin0, end0, data0);
            auto data0Squeezed = xtensor::squeezedView(data0);

            vigra::HistogramOptions histoOptions;
            auto minMax = std::minmax_element(data0Squeezed.begin(), data0Squeezed.end());
            float min = static_cast<float>(*(minMax.first));
            float max = static_cast<float>(*(minMax.second));
            histoOptions.setMinMax(min, max);

            // construct slice pairs for processing in parallel
            std::vector<std::pair<int64_t,int64_t>> slicePairs;
            int64_t lowerSliceId = 0;
            int64_t upperSliceId = 1;
            while(upperSliceId < numberOfSlices) {
                slicePairs.emplace_back(std::make_pair(lowerSliceId,upperSliceId));
                ++lowerSliceId;
                ++upperSliceId;
            }

            // process slice pairs in parallel
            parallel::parallel_foreach(threadpool, slicePairs.size(), [&](const int tid, const int64_t pairId){

                std::cout << "Processing slice pair: " << pairId << " / " << slicePairs.size() << std::endl;
                int64_t sliceIdA = slicePairs[pairId].first; // lower slice
                int64_t sliceIdB = slicePairs[pairId].second;// upper slice

                // compute the filters for slice A
                Coord beginA ({sliceIdA, 0L, 0L});
                Coord endA({sliceIdA+1, shape[1], shape[2]});

                auto labelsA = labelsAStorage.getView(tid);
                tools::readSubarray(labels, beginA, endA, labelsA);
                auto labelsASqueezed = xtensor::squeezedView(labelsA);

                auto dataA = dataAStorage.getView(tid);
                tools::readSubarray(data, beginA, endA, dataA);
                auto dataASqueezed = xtensor::squeezedView(dataA);

                // copy the data if our input is not float
                auto dataACopy = dataACopyStorage.getView(tid);
                copyIfNecessary(dataASqueezed, dataACopy, sliceShape2);

                // acccumulate the inner slice features
                // only if not keepZOnly and if we have at least one edge in this slice
                // (no edge can happend for defected slices)
                if( rag.numberOfInSliceEdges(sliceIdA) > 0 && !keepZOnly) {
                    auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdA);
                    // resize the current acc chain vector
                    EdgeAccChainVectorType accChainVec(rag.numberOfInSliceEdges(sliceIdA));
                    accumulateInnerSliceFeatures(
                        accChainVec,
                        histoOptions,
                        sliceShape2,
                        labelsASqueezed,
                        sliceIdA,
                        inEdgeOffset,
                        rag,
                        dataACopy,
                        pass
                    );
                    fXY(accChainVec, sliceIdA, inEdgeOffset);
                }

                //
                // process upper slice
                // do if we are not keeping only xy edges or
                // if we are at the last slice (which is never a lower slice and
                // must hence be accumulated extra)
                //
                if(!keepXYOnly || sliceIdB == numberOfSlices - 1 ) {

                    // init labels and data for upper slice
                    Coord beginB = Coord({sliceIdB,   0L,       0L});
                    Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});
                    auto dataBCopy = dataBCopyStorage.getView(tid);

                    // read labels
                    auto labelsB = labelsBStorage.getView(tid);
                    tools::readSubarray(labels, beginB, endB, labelsB);
                    auto labelsBSqueezed = xtensor::squeezedView(labelsB);
                    // read data
                    auto dataB = dataBStorage.getView(tid);
                    tools::readSubarray(data, beginB, endB, dataB);
                    auto dataBSqueezed = xtensor::squeezedView(dataB);

                    copyIfNecessary(dataBSqueezed, dataBCopy, sliceShape2);

                    // acccumulate the between slice features
                    if(!keepXYOnly) {
                        auto betweenEdgeOffset = rag.betweenSliceEdgeOffset(sliceIdA);
                        auto accOffset = rag.betweenSliceEdgeOffset(sliceIdA) - rag.numberOfInSliceEdges();
                        // resize the current acc chain vector
                        EdgeAccChainVectorType accChainVec(rag.numberOfInBetweenSliceEdges(sliceIdA));
                        // accumulate features for the in between slice edges
                        accumulateBetweenSliceFeatures(accChainVec,
                                                       histoOptions,
                                                       sliceShape2,
                                                       labelsASqueezed,
                                                       labelsBSqueezed,
                                                       sliceIdA,
                                                       sliceIdB,
                                                       betweenEdgeOffset,
                                                       rag,
                                                       dataACopy,
                                                       dataBCopy,
                                                       zDirection,
                                                       pass);
                        fZ(accChainVec, sliceIdA, accOffset);
                    }

                    // accumulate the inner slice features for the last slice, which is never a lower slice
                    if(!keepZOnly && (sliceIdB == numberOfSlices - 1 && rag.numberOfInSliceEdges(sliceIdB) > 0)) {
                        auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdB);
                        // resize the current acc chain vector
                        EdgeAccChainVectorType accChainVec(rag.numberOfInSliceEdges(sliceIdB));
                        accumulateInnerSliceFeatures(accChainVec,
                                                     histoOptions,
                                                     sliceShape2,
                                                     labelsBSqueezed,
                                                     sliceIdB,
                                                     inEdgeOffset,
                                                     rag,
                                                     dataBCopy,
                                                     pass);
                        fXY(accChainVec, sliceIdB, inEdgeOffset);
                    }
                }

            });
        }
    }


    // 9 features
    template<class LABELS_PROXY, class DATA, class OUTPUT>
    void accumulateEdgeStandardFeatures(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        const DATA & data,
        OUTPUT & edgeFeaturesOutXY,
        OUTPUT & edgeFeaturesOutZ,
        const bool keepXYOnly,
        const bool keepZOnly,
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
        typedef array::StaticArray<int64_t, 2> FeatCoord;

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

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
            auto accFunction = [](const std::vector<AccChainType> & edgeAccChainVec,
                                  const size_t sliceId,
                                  const size_t edgeOffset,
                                  OUTPUT & edgeFeaturesOut,
                                  OverhangStorage & storageFront,
                                  OverhangStorage & storageBack){
                using namespace vigra::acc;

                const auto edgeChunkSize = tools::getChunkShape(edgeFeaturesOut)[0];

                const auto nEdges = edgeAccChainVec.size();
                const auto nFeats = 9;

                xt::xtensor<DataType, 2> featuresTemp({nEdges, nFeats});
                for(auto edge = 0; edge < edgeAccChainVec.size(); ++edge) {
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    featuresTemp(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    featuresTemp(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        featuresTemp(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }

                const int64_t edgeEnd = edgeOffset + nEdges;
                const int64_t overhangBegin = (edgeOffset % edgeChunkSize == 0) ? 0 : edgeChunkSize - (edgeOffset % edgeChunkSize);
                const int64_t overhangEnd = edgeEnd % edgeChunkSize;
                // Your engine's dead, there's something wrong; can you hear me Major Tom?

                // find beginning and end for block-aligned edges in tmp features
                // at the begin, we need to check
                const int64_t edgeEndAlignedLocal = nEdges - overhangEnd;
                FeatCoord beginAlignedLocal{(int64_t)overhangBegin, 0L};
                FeatCoord endAlignedLocal{(int64_t)edgeEndAlignedLocal, (int64_t)nFeats};

                // write the aligned features - if any exist
                if(edgeEndAlignedLocal > 0) {
                    // get view to the aligned features
                    xt::slice_vector sliceAligned(featuresTemp);
                    xtensor::sliceFromRoi(sliceAligned, beginAlignedLocal, endAlignedLocal);
                    auto featuresAligned = xt::dynamic_view(featuresTemp, sliceAligned);

                    // find global beginning and end for block aligned edges
                    FeatCoord beginAlignedGlobal{int64_t(edgeOffset + overhangBegin), 0L};
                    FeatCoord endAlignedGlobal{int64_t(edgeEnd - overhangEnd), (int64_t)nFeats};

                    // write out blockaligned edges
                    tools::writeSubarray(edgeFeaturesOut, beginAlignedGlobal, endAlignedGlobal, featuresAligned);
                }

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

            accumulateEdgeFeaturesWithAccChain<AccChainType>(rag,
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
                writeOverhangingChunks(storageXYFront, storageXYBack, edgeFeaturesOutXY, threadpool, false);
            }

            if(!keepXYOnly) {
                writeOverhangingChunks(storageZFront, storageZBack, edgeFeaturesOutZ, threadpool, true);
            }

        } else {

            // accumulator function
            // no chunk support
            auto accFunction = [](const std::vector<AccChainType> & edgeAccChainVec,
                                  const size_t sliceId,
                                  const uint64_t edgeOffset,
                                  OUTPUT & edgeFeaturesOut){

                using namespace vigra::acc;

                const uint64_t nEdges = edgeAccChainVec.size();
                const uint64_t nFeats = 9;

                xt::xtensor<DataType, 2> featuresTemp({nEdges, nFeats});

                for(auto edge = 0; edge < edgeAccChainVec.size(); ++edge) {
                    const auto & chain = edgeAccChainVec[edge];
                    const auto mean = get<acc::Mean>(chain);
                    const auto quantiles = get<Quantiles>(chain);
                    featuresTemp(edge, 0) = replaceIfNotFinite(mean,     0.0);
                    featuresTemp(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    for(auto qi=0; qi<7; ++qi)
                        featuresTemp(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                }

                FeatCoord begin({int64_t(edgeOffset),0L});
                FeatCoord end({edgeOffset+nEdges, nFeats});

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

            accumulateEdgeFeaturesWithAccChain<AccChainType>(rag,
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


    /*
    // FIXME we rely on the out vector coming in the right shape from python,
    // that is kindof hacky...
    template<class LABELS_PROXY>
    void getSkipEdgeLengths(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        std::vector<size_t> & out, // TODO call by ref or call by val ?
        const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
        const std::vector<size_t> & skipRanges,
        const std::vector<size_t> & skipStarts,
        const int numberOfThreads = -1
    ){
        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        getSkipEdgeLengthsImpl(rag, out, skipEdges, skipRanges, skipStarts, pOpts, threadpool);
    }


    template<class LABELS_PROXY>
    void getSkipEdgeLengthsImpl(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        std::vector<size_t> & out,
        const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
        const std::vector<size_t> & skipRanges,
        const std::vector<size_t> & skipStarts,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool
    ){
        typedef LABELS_PROXY LabelsProxyType;

        typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        typedef std::pair<uint64_t,uint64_t> SkipEdgeStorage;
        typedef std::vector<size_t> EdgeLenVector;

        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        const auto & shape = rag.shape();
        const auto & labelsProxy = rag.labelsProxy();

        Coord2 sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({1L,shape[1], shape[2]});

        LabelBlockStorage labelsAStorage(threadpool, sliceShape3, 1);
        LabelBlockStorage labelsBStorage(threadpool, sliceShape3, 1);

        // get unique lower slices with skip edges
        std::vector<size_t> lowerSlices;
        tools::uniques(skipStarts, lowerSlices);
        auto lowest = int64_t(lowerSlices[0]);

        // get upper slices with skip edges for each lower slice and number of skip edges for each lower slice
        std::map<size_t,std::vector<size_t>> skipSlices;
        std::map<size_t,size_t> numberOfSkipEdgesPerSlice;
        // initialize the maps
        for(auto sliceId : lowerSlices) {
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

        int countSlice = 0;
        size_t skipEdgeOffset = 0;
        for(auto sliceId : lowerSlices) {

            std::cout << countSlice++ << " / " << lowerSlices.size() << std::endl;
            std::cout << "Computing lengths for skip edges from slice " << sliceId << std::endl;

            Coord beginA({int64_t(sliceId),0L,0L});
            Coord endA(  {int64_t(sliceId+1),shape[1],shape[2]});
            auto labelsA = labelsAStorage.getView(0);
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            auto skipEdgesInSlice = numberOfSkipEdgesPerSlice[sliceId];
            std::vector<EdgeLenVector> perThreadData(actualNumberOfThreads, EdgeLenVector(skipEdgesInSlice));

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

                // accumulate filter for the between slice edges
                nifty::tools::parallelForEachCoordinate(threadpool, sliceShape2, [&](const int tid, const Coord2 coord){

                    auto & threadData = perThreadData[tid];

                    // labels are different for different slices by default!
                    const auto lU = xtensor::read(labelsASqueezed, coord.asStdArray());
                    const auto lV = xtensor::read(labelsBSqueezed, coord.asStdArray());

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
                    ++threadData[skipId];

                });
            }

            // write the thread data into out in parallel over the edges
            parallel::parallel_foreach(threadpool, skipEdgesInSlice, [&](const int tid, const int64_t skipId){

                auto & outData = out[skipId + skipEdgeOffset];
                for(size_t t = 0; t < actualNumberOfThreads; ++t) {
                    auto & threadData = perThreadData[t];
                    outData += threadData[skipId];
                }

            });
            skipEdgeOffset += skipEdgesInSlice;
        }
    }
    */


} // end namespace graph
} // end namespace nifty
