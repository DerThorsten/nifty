#pragma once

#include "nifty/graph/rag/grid_rag_accumulate.hxx"

// accumulate features for flat superpixels with normal rag

namespace nifty{
namespace graph{

template<class ACC_CHAIN_VECTOR, class COORD, class LABEL_ARRAY, class DATA_ARRAY, class RAG>
inline void accumulateInnerSliceFeatures(ACC_CHAIN_VECTOR & accChainVec,
                                         const COORD & sliceShape2,
                                         const xt::xexpression<LABEL_ARRAY> & labelsSqueezedExp,
                                         const RAG & rag,
                                         const xt::xexpression<DATA_ARRAY> & dataExp,
                                         const int pass,
                                         const int64_t sliceId) {

    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    const auto & labelsSqueezed = labelsSqueezedExp.derived_cast();
    const auto & data = dataExp.derived_cast();

    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        const auto lU = xtensor::read(labelsSqueezed, coord.asStdArray());
        for(int axis = 0; axis < 2; ++axis){
            Coord2 coord2 = coord;
            ++coord2[axis];
            if( coord2[axis] < sliceShape2[axis]) {
                const auto lV = xtensor::read(labelsSqueezed, coord2.asStdArray());
                if(lU != lV) {
                    VigraCoord vigraCoordU;
                    VigraCoord vigraCoordV;
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = sliceId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord2[d-1];
                    }
                    const auto edge = rag.findEdge(lU,lV);
                    const auto fU = xtensor::read(data, coord.asStdArray());
                    const auto fV = xtensor::read(data, coord2.asStdArray());
                    accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                    accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
                }
            }
        }
    });
}

template<class ACC_CHAIN_VECTOR, class COORD, class LABEL_ARRAY, class DATA_ARRAY, class RAG>
inline void accumulateBetweenSliceFeatures(ACC_CHAIN_VECTOR & accChainVec,
                                           const COORD & sliceShape2,
                                           const xt::xexpression<LABEL_ARRAY> & labelsASqueezedExp,
                                           const xt::xexpression<LABEL_ARRAY> & labelsBSqueezedExp,
                                           const RAG & rag,
                                           const xt::xexpression<DATA_ARRAY> & dataAExp,
                                           const xt::xexpression<DATA_ARRAY> & dataBExp,
                                           const int pass,
                                           const int64_t sliceIdA,
                                           const int64_t sliceIdB,
                                           const int zDirection = 0){

    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    const auto &labelsASqueezed = labelsASqueezedExp.derived_cast();
    const auto &labelsBSqueezed = labelsBSqueezedExp.derived_cast();
    
    const auto &dataA = dataAExp.derived_cast();
    const auto &dataB = dataBExp.derived_cast();

    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        const auto lU = xtensor::read(labelsASqueezed, coord.asStdArray());
        const auto lV = xtensor::read(labelsBSqueezed, coord.asStdArray());
        if(lU != lV) {
            VigraCoord vigraCoordU;
            VigraCoord vigraCoordV;
            vigraCoordU[0] = sliceIdA;
            vigraCoordV[0] = sliceIdB;
            for(int d = 1; d < 3; ++d){
                vigraCoordU[d] = coord[d-1];
                vigraCoordV[d] = coord[d-1];
            }
            const auto edge = rag.findEdge(lU,lV);
            if(zDirection==0) { // 0 -> take into account z and z + 1
                const auto fU = xtensor::read(dataA, coord.asStdArray());
                const auto fV = xtensor::read(dataB, coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
            else if(zDirection==1) { // 1 -> take into accout only z
                const auto fU = xtensor::read(dataA, coord.asStdArray());
                accChainVec[edge].updatePassN(fU, vigraCoordU, pass);
            }
            else if(zDirection==2) { // 2 -> take into accout only z + 1
                const auto fV = xtensor::read(dataB, coord.asStdArray());
                accChainVec[edge].updatePassN(fV, vigraCoordV, pass);
            }
        }
    });
}


template<class EDGE_ACC_CHAIN, class LABELS, class DATA, class F>
void accumulateEdgeFeaturesFlatWithAccChain(
    const GridRag<3, LABELS> & rag,
    const DATA & data,
    const parallel::ParallelOptions & pOpts,
    parallel::ThreadPool & threadpool,
    F && f,
    const AccOptions & accOptions = AccOptions()
){
    typedef LABELS LabelsType;
    typedef typename LabelsType::value_type LabelType;
    typedef typename DATA::value_type DataType;

    typedef typename GridRag<3, LabelsType>::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;

    const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labels = rag.labels();

    uint64_t numberOfSlices = shape[0];

    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({static_cast<int64_t>(1), shape[1], shape[2]});

    // edge acc vectors for multiple threads
    std::vector<AccChainVectorType> perThreadAccChainVector(actualNumberOfThreads);
    parallel::parallel_foreach(threadpool, actualNumberOfThreads,
    [&](const int tid, const int64_t i){
        perThreadAccChainVector[i] = AccChainVectorType(rag.edgeIdUpperBound()+1);
    });

    if(accOptions.setMinMax){
        vigra::HistogramOptions histogram_opt;
        histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal);
        parallel::parallel_foreach(threadpool, actualNumberOfThreads,
        [&](int tid, int i){
            auto & edgeAccVec = perThreadAccChainVector[i];
            for(auto & edgeAcc : edgeAccVec){
                edgeAcc.setHistogramOptions(histogram_opt);
            }
        });
    }

    // construct slice pairs for processing in parallel
    std::vector<std::pair<int64_t,int64_t>> slicePairs;
    int64_t lowerSliceId = 0;
    int64_t upperSliceId = 1;
    while(upperSliceId < numberOfSlices) {
        slicePairs.emplace_back(std::make_pair(lowerSliceId,upperSliceId));
        ++lowerSliceId;
        ++upperSliceId;
    }


    const auto passesRequired = (perThreadAccChainVector.front()).front().passesRequired();
    for(auto pass = 1; pass <= passesRequired; ++pass) {
        // std::cout << "Pass " << pass << " / " << passesRequired << std::endl;

        // label and data storages
        LabelBlockStorage  labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage  labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        DataBlockStorage   dataAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        DataBlockStorage   dataBStorage(threadpool, sliceShape3, actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, slicePairs.size(), [&](const int tid, const int64_t pairId){

            //std::cout << "Processing slice pair: " << pairId << " / " << slicePairs.size() << std::endl;
            int64_t sliceIdA = slicePairs[pairId].first; // lower slice
            int64_t sliceIdB = slicePairs[pairId].second;// upper slice
            //std::cout << "Upper: " << sliceIdA << " Lower: " << sliceIdB << std::endl;
            auto & threadAccChainVec = perThreadAccChainVector[tid];

            Coord beginA ({sliceIdA, static_cast<int64_t>(0), static_cast<int64_t>(0)});
            Coord endA({sliceIdA+1, shape[1], shape[2]});

            auto labelsA = labelsAStorage.getView(tid);
            tools::readSubarray(labels, beginA, endA, labelsA);
            auto labelsASqueezed = xtensor::squeezedView(labelsA);

            auto dataA = dataAStorage.getView(tid);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = xtensor::squeezedView(dataA);

            accumulateInnerSliceFeatures(
                threadAccChainVec,
                sliceShape2,
                labelsASqueezed,
                rag,
                dataASqueezed,
                pass,
                sliceIdA
            );

            // process upper slice
            Coord beginB = Coord({sliceIdB, static_cast<int64_t>(0), static_cast<int64_t>(0)});
            Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});

            // read labels and data for upper slice
            auto labelsB = labelsBStorage.getView(tid);
            tools::readSubarray(labels, beginB, endB, labelsB);
            auto labelsBSqueezed = xtensor::squeezedView(labelsB);
            auto dataB = dataBStorage.getView(tid);
            tools::readSubarray(data, beginB, endB, dataB);
            auto dataBSqueezed = xtensor::squeezedView(dataB);

            // accumulate features for the in between slice edges
            accumulateBetweenSliceFeatures(
                    threadAccChainVec,
                    sliceShape2,
                    labelsASqueezed,
                    labelsBSqueezed,
                    rag,
                    dataASqueezed,
                    dataBSqueezed,
                    pass,
                    sliceIdA,
                    sliceIdB,
                    accOptions.zDirection);

            // accumulate the inner slice features for the last slice,
            // which is never a lower slice
            if(sliceIdB == numberOfSlices - 1) {
                accumulateInnerSliceFeatures(
                        threadAccChainVec,
                        sliceShape2,
                        labelsBSqueezed,
                        rag,
                        dataBSqueezed,
                        pass,
                        sliceIdB);
            }
        });
    }

    // merge the accumulators in parallel
    auto & resultAccVec = perThreadAccChainVector.front();
    parallel::parallel_foreach(threadpool, resultAccVec.size(),
    [&](const int tid, const int64_t edge){
        for(auto t=1; t<actualNumberOfThreads; ++t){
            resultAccVec[edge].merge((perThreadAccChainVector[t])[edge]);
        }
    });
    // call functor with finished acc chain
    f(resultAccVec);
}


// 9 features
template<std::size_t DIM, class LABELS, class DATA, class FEATURE_TYPE>
void accumulateEdgeFeaturesFlat(
    const GridRag<DIM, LABELS> & rag,
    const DATA & data,
    const double minVal,
    const double maxVal,
    xt::xexpression<FEATURE_TYPE> & edgeFeaturesOutExp,
    const int zDirection = 0,
    const int numberOfThreads = -1
){
    namespace acc = vigra::acc;
    typedef typename FEATURE_TYPE::value_type DataType;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    // FIXME skewness and kurtosis are broken
    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        //acc::Skewness,    //1
        //acc::Kurtosis,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<DIM, DataType, SelectType> AccChainType;

    auto & edgeFeaturesOut = edgeFeaturesOutExp.derived_cast();

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);
    const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    accumulateEdgeFeaturesFlatWithAccChain<AccChainType>(
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
                //edgeFeaturesOut(edge, 2) = replaceIfNotFinite(get<acc::Skewness>(chain), 0.0);
                //edgeFeaturesOut(edge, 3) = replaceIfNotFinite(get<acc::Kurtosis>(chain), 0.0);
                for(auto qi=0; qi<7; ++qi)
                    edgeFeaturesOut(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
            });
        },
        AccOptions(minVal, maxVal, zDirection)
    );
}

} // namespace graph
} // namespace nifty
