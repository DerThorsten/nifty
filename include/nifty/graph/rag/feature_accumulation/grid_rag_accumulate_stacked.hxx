#pragma once

#include <vector>
#include <cmath>

#include "nifty/tools/array_tools.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

namespace nifty{
namespace graph{

    template<class ACC_CHAIN_VECTOR, class HISTO_OPTS, class COORD, class LABEL_TYPE, class RAG>
    inline void accumulateInnerSliceFeatures(
            ACC_CHAIN_VECTOR & accChainVec,
            const HISTO_OPTS & histoOptions,
            const COORD & sliceShape2,
            const marray::View<LABEL_TYPE> & labels,
            const int64_t sliceId,
            const int64_t inEdgeOffset,
            const RAG & rag,
            const marray::View<float> & data,
            const int pass
        ) {

        typedef COORD Coord2;
        typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

        // set minmax for accumulator chains
        for(int64_t edge = 0; edge < accChainVec.size(); ++edge){
            accChainVec[edge].setHistogramOptions(histoOptions);
        }

        // accumulate filter for the inner slice edges
        nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
            const auto lU = labels(coord.asStdArray());
            for(int axis = 0; axis < 2; ++axis){
                Coord2 coord2 = coord;
                ++coord2[axis];
                if( coord2[axis] < sliceShape2[axis]) {
                    const auto lV = labels(coord2.asStdArray());
                    if(lU != lV) {
                        VigraCoord vigraCoordU, vigraCoordV;
                        vigraCoordU[0] = sliceId;
                        vigraCoordV[0] = sliceId;
                        for(int d = 1; d < 3; ++d){
                            vigraCoordU[d] = coord[d-1];
                            vigraCoordV[d] = coord2[d-1];
                        }
                        const auto edge = rag.findEdge(lU,lV) - inEdgeOffset;
                        const auto fU = data(coord.asStdArray());
                        const auto fV = data(coord2.asStdArray());
                        accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                        accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
                    }
                }
            }
        });
    }


    // accumulate filter for the between slice edges
    template<class ACC_CHAIN_VECTOR, class HISTO_OPTS, class COORD, class LABEL_TYPE, class RAG>
    inline void accumulateBetweenSliceFeatures(
            ACC_CHAIN_VECTOR & accChainVec,
            const HISTO_OPTS & histoOptions,
            const COORD & sliceShape2,
            const marray::View<LABEL_TYPE> & labelsA,
            const marray::View<LABEL_TYPE> & labelsB,
            const int64_t sliceIdA,
            const int64_t sliceIdB,
            const int64_t betweenEdgeOffset,
            const RAG & rag,
            const marray::View<float> & dataA,
            const marray::View<float> & dataB,
            const int zDirection,
            const int pass
        ){

        typedef COORD Coord2;
        typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

        // set minmax for accumulator chains
        for(int64_t edge = 0; edge < accChainVec.size(); ++edge){
            accChainVec[edge].setHistogramOptions(histoOptions);
        }

        nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
            // labels are different for different slices by default!
            const auto lU = labelsA(coord.asStdArray());
            const auto lV = labelsB(coord.asStdArray());
            VigraCoord vigraCoordU, vigraCoordV;
            vigraCoordU[0] = sliceIdA;
            vigraCoordV[0] = sliceIdB;
            for(int d = 1; d < 3; ++d){
                vigraCoordU[d] = coord[d-1];
                vigraCoordV[d] = coord[d-1];
            }
            const auto edge = rag.findEdge(lU,lV) - betweenEdgeOffset;
            if(zDirection==0) { // 0 -> take into account z and z + 1
                const auto fU = dataA(coord.asStdArray());
                const auto fV = dataB(coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
            else if(zDirection==1) { // 1 -> take into accout only z
                const auto fU = dataA(coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
            }
            else if(zDirection==2) { // 2 -> take into accout only z + 1
                const auto fV = dataB(coord.asStdArray());
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
        });
    }


    // copy data if the dtype is not float
    template<class DATA_TYPE, class COORD>
    inline void copyIfNecessary(
        const marray::View<DATA_TYPE> & dataSqueezed,
        marray::View<float> & dataCopy,
        const COORD & sliceShape2
    ) {

        typedef DATA_TYPE DataType;
        typedef COORD Coord;
        marray::View<float> dataView;
        if( typeid(DataType) == typeid(float) ) {
            dataCopy = dataSqueezed;
            return;
        }
        else {
            // copy the data (we don't use std::copy here, because iterators are terribly
            // slow for marrays)
            tools::forEachCoordinate(
                sliceShape2, [&dataCopy, &dataSqueezed](Coord coord){
                    dataCopy(coord.asStdArray()) = (float) dataSqueezed(coord.asStdArray());
            });
            return;
        }
    }


    // accumulator with data
    template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F_XY, class F_Z>
    void accumulateEdgeFeaturesWithAccChain(
        const GridRagStacked2D<LABELS_PROXY> & rag,
        const DATA & data,
        const bool keepXYOnly,
        const bool keepZOnly,
        const parallel::ParallelOptions & pOpts,
        parallel::ThreadPool & threadpool,
        F_XY && fXY,
        F_Z && fZ,
        const int zDirection
    ){

        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LabelsProxyType::LabelType LabelType;
        typedef typename DATA::DataType DataType;

        typedef typename vigra::MultiArrayShape<3>::type   VigraCoord;
        typedef typename LabelsProxyType::BlockStorageType LabelStorage;
        typedef typename tools::BlockStorageSelector<DATA>::type DataStorage;
        typedef tools::BlockStorage<float> DataCopyStorage;

        typedef array::StaticArray<int64_t,3> Coord;
        typedef array::StaticArray<int64_t,2> Coord2;
        typedef EDGE_ACC_CHAIN EdgeAccChainType;
        typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();

        EdgeAccChainVectorType edgeAccChainVector(rag.edgeIdUpperBound()+1);

        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        const Coord2 sliceShape2({shape[1], shape[2]});
        const Coord  sliceShape3({1L,shape[1], shape[2]});

        const auto passesRequired = edgeAccChainVector.front().passesRequired();

        // For now, we only support single pass!
        // do N passes of accumulator
        //for(auto pass=1; pass <= passesRequired; ++pass){
        int pass = 1;
        {
            // accumulate inner slice feature
            // edge acc vectors for multiple threads
            std::vector<EdgeAccChainVectorType> perThreadAccChainVector(nThreads);

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
            auto data0Squeezed = data0.squeezedView();

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
                auto & accChainVec = perThreadAccChainVector[tid];

                // compute the filters for slice A
                Coord beginA ({sliceIdA, 0L, 0L});
                Coord endA({sliceIdA+1, shape[1], shape[2]});

                auto labelsA = labelsAStorage.getView(tid);
                labelsProxy.readSubarray(beginA, endA, labelsA);
                auto labelsASqueezed = labelsA.squeezedView();

                auto dataA = dataAStorage.getView(tid);
                tools::readSubarray(data, beginA, endA, dataA);
                auto dataASqueezed = dataA.squeezedView();

                // copy the data if our input is not float
                auto dataACopy = dataACopyStorage.getView(tid);
                copyIfNecessary(dataASqueezed, dataACopy, sliceShape2);

                // acccumulate the inner slice features
                // only if not keepZOnly and if we have at least one edge in this slice
                // (no edge can happend for defected slices)
                if( rag.numberOfInSliceEdges(sliceIdA) > 0 && !keepZOnly) {
                    auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdA);
                    // resize the current acc chain vector
                    accChainVec = EdgeAccChainVectorType(rag.numberOfInSliceEdges(sliceIdA));
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
                    fXY(accChainVec, inEdgeOffset);
                }

                // process upper slice
                // TODO copy non-float data ?!
                Coord beginB = Coord({sliceIdB,   0L,       0L});
                Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});
                marray::View<LabelType> labelsBSqueezed;
                marray::View<DataType> dataBSqueezed;
                auto dataBCopy = dataBCopyStorage.getView(tid);

                // read labels and data for upper slice
                // do if we are not keeping only xy edges or
                // if we are at the last slice (which is never a lower slice and
                // must hence be accumulated extra)
                if(!keepXYOnly || sliceIdB == numberOfSlices - 1 ) {
                    // read labels
                    auto labelsB = labelsBStorage.getView(tid);
                    labelsProxy.readSubarray(beginB, endB, labelsB);
                    labelsBSqueezed = labelsB.squeezedView();
                    // read data
                    auto dataB = dataBStorage.getView(tid);
                    tools::readSubarray(data, beginB, endB, dataB);
                    dataBSqueezed = dataB.squeezedView();

                    copyIfNecessary(dataBSqueezed, dataBCopy, sliceShape2);
                }

                // acccumulate the between slice features
                if(!keepXYOnly) {
                    auto betweenEdgeOffset = rag.betweenSliceEdgeOffset(sliceIdA);
                    auto accOffset = rag.betweenSliceEdgeOffset(sliceIdA) - rag.numberOfInSliceEdges();
                    // resize the current acc chain vector
                    accChainVec = EdgeAccChainVectorType(rag.numberOfInBetweenSliceEdges(sliceIdA));
                    // accumulate features for the in between slice edges
                    accumulateBetweenSliceFeatures(
                        accChainVec,
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
                        pass
                    );
                    fZ(accChainVec, accOffset);
                }

                // accumulate the inner slice features for the last slice, which is never a lower slice
                if(!keepZOnly && (sliceIdB == numberOfSlices - 1 && rag.numberOfInSliceEdges(sliceIdB) > 0)) {
                    auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdB);
                    // resize the current acc chain vector
                    accChainVec = EdgeAccChainVectorType(rag.numberOfInSliceEdges(sliceIdB));
                    accumulateInnerSliceFeatures(
                            accChainVec,
                            histoOptions,
                            sliceShape2,
                            labelsBSqueezed,
                            sliceIdB,
                            inEdgeOffset,
                            rag,
                            dataBCopy,
                            pass
                    );
                    fXY(accChainVec, inEdgeOffset);
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

        // threadpool
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

        // general accumulator function
        auto accFunction = [&threadpool](
            const std::vector<AccChainType> & edgeAccChainVec,
            const uint64_t edgeOffset,
            OUTPUT & edgeFeaturesOut
        ){
            using namespace vigra::acc;
            typedef array::StaticArray<int64_t, 2> FeatCoord;

            const uint64_t nEdges = edgeAccChainVec.size();
            const uint64_t nStats = 9;

            marray::Marray<DataType> featuresTemp({nEdges, nStats});

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
            FeatCoord end({edgeOffset+nEdges, nStats});

            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        };

        // instantiation of accumulators for xy / z edges
        auto accFunctionXY = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::ref(edgeFeaturesOutXY));
        auto accFunctionZ = std::bind(accFunction,
                std::placeholders::_1,
                std::placeholders::_2,
                std::ref(edgeFeaturesOutZ));

        accumulateEdgeFeaturesWithAccChain<AccChainType>(
            rag,
            data,
            keepXYOnly,
            keepZOnly,
            pOpts,
            threadpool,
            accFunctionXY,
            accFunctionZ,
            zDirection
        );
    }


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


} // end namespace graph
} // end namespace nifty
