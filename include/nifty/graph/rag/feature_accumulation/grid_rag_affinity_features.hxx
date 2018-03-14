#pragma once

#include "nifty/graph/rag/grid_rag_accumulate.hxx"
// lifted extraction is not supported for now
// #include "nifty/graph/rag/feature_accumulation/lifted_nh.hxx"


namespace nifty{
namespace graph{


template<class EDGE_ACC_CHAIN, class RAG, class AFFINITIES, class F>
void accumulateAffninitiesWithAccChain(const RAG & rag,
                                       const AFFINITIES & affinities,
                                       const std::vector<std::array<int, 3>> & offsets,
                                       parallel::ThreadPool & threadpool,
                                       F && f,
                                       const AccOptions & accOptions = AccOptions()){
    typedef array::StaticArray<int64_t, 3> Coord3;
    typedef array::StaticArray<int64_t, 4> Coord4;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType> AccChainVectorType;
    typedef std::vector<AccChainVectorType> ThreadAccChainVectorType;

    const auto & labels = rag.labels();

    Coord3 shape;
    Coord4 affShape;
    affShape[0] = affinities.shape()[0];
    for(int d = 0; d < 3; ++d) {
        shape[d] = labels.shape()[d];
        affShape[d+1] = affinities.shape()[d+1];
    }

    // only single threaded for now
    // accumulator chain vectors for local and lifted edges
    size_t nEdges = rag.edgeIdUpperBound() + 1;
    auto nThreads = threadpool.nThreads();
    ThreadAccChainVectorType edgeAccumulators(nThreads);

    vigra::HistogramOptions histogram_opt;
    histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal);

    parallel::parallel_foreach(threadpool, nThreads,
    [&](int tid, int threadId){
        auto & thisAccumulators = edgeAccumulators[threadId];
        thisAccumulators = AccChainVectorType(nEdges);
        // set the histogram options
        if(accOptions.setMinMax){
            for(size_t edgeId; edgeId < nEdges; ++edgeId) {
                thisAccumulators[edgeId].setHistogramOptions(histogram_opt);
            }
        }
    });

    int pass = 1;

    // iterate over all affinity links and accumulate the associated
    // affinity edges
    tools::parallelForEachCoordinate(threadpool, affShape, [&](int tid, const Coord4 & affCoord) {

        Coord3 cU, cV;
        VigraCoord vc;
        const auto & offset = offsets[affCoord[0]];

        for(int d = 0; d < 3; ++d) {
            cU[d] = affCoord[d+1];
            cV[d] = affCoord[d+1] + offset[d];
            // range check
            if(cV[d] < 0 || cV[d] >= shape[d]) {
                return;
            }
        }

        const auto u = xtensor::read(labels, cU.asStdArray());
        const auto v = xtensor::read(labels, cV.asStdArray());

        // only do stuff if the labels are different
        if(u != v) {

            auto & thisAccumulators = edgeAccumulators[tid];
            // we just update the vigra coord of label u
            for(int d = 0; d < 3; ++d) {
                vc = cU[d];
            }

            const auto val = xtensor::read(affinities, affCoord.asStdArray());
            const int64_t e = rag.findEdge(u, v);
            // For long range affinities, edge might not be in the rag
            if(e != -1) {
                thisAccumulators[e].updatePassN(val, vc, pass);
            }
        }
    });

    // merge accumulators
    auto & resultAccVec = edgeAccumulators.front();
    parallel::parallel_foreach(threadpool, resultAccVec.size(),
    [&](const int tid, const int64_t edge){
        for(auto t=1; t<nThreads; ++t){
            resultAccVec[edge].merge((edgeAccumulators[t])[edge]);
        }
    });

    f(resultAccVec);
}


// 9 features
template<class RAG, class AFFINITIES, class FEATURE_ARRAY>
void accumulateAffinities(
    const RAG & rag,
    const AFFINITIES & affinities,
    const std::vector<std::array<int, 3>> & offsets,
    xt::xexpression<FEATURE_ARRAY> & featuresExp,
    const double minVal = 0.,
    const double maxVal = 1.,
    const int numberOfThreads = -1
){
    // check that shapes off affs and labels agree
    namespace acc = vigra::acc;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    // TODO need to accumulate edge size here as well !!!
    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, double, SelectType> AccChainType;

    auto & features = featuresExp.derived_cast();

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);
    const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    auto accumulate = [&](
        const std::vector<AccChainType> & edgeAccChainVec
    ){
        using namespace vigra::acc;

        parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
            const int tid, const int64_t edge
        ){
            const auto & chain = edgeAccChainVec[edge];
            const auto mean = get<acc::Mean>(chain);
            const auto quantiles = get<Quantiles>(chain);
            features(edge, 0) = replaceIfNotFinite(mean,     0.0);
            features(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
            for(auto qi=0; qi<7; ++qi)
                features(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
        });
    };

    accumulateAffninitiesWithAccChain<AccChainType>(rag,
                                                    affinities,
                                                    offsets,
                                                    threadpool,
                                                    accumulate,
                                                    AccOptions(minVal, maxVal));
}


// lifted extraction is not supported for now
template<class EDGE_ACC_CHAIN, class RAG, class LNH, class AFFINITIES, class F_LOCAL, class F_LIFTED>
void accumulateLongRangeAffninitiesWithAccChain(const RAG & rag,
                                                const LNH & lnh,
                                                const AFFINITIES & affinities,
                                                parallel::ThreadPool & threadpool,
                                                F_LOCAL && f_local,
                                                F_LIFTED && f_lifted,
                                                const AccOptions & accOptions = AccOptions()){
    typedef array::StaticArray<int64_t, 3> Coord3;
    typedef array::StaticArray<int64_t, 4> Coord4;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;
    typedef std::vector<AccChainVectorType> ThreadAccChainVectorType;

    const auto & labels = rag.labels();

    Coord3 shape;
    Coord4 affShape;
    affShape[0] = affinities.shape()[0];
    for(int d = 0; d < 3; ++d) {
        shape[d] = labels.shape()[d];
        affShape[d+1] = shape[d];
    }

    size_t nLocal = rag.edgeIdUpperBound() + 1;
    size_t nLifted = lnh.edgeIdUpperBound() + 1;
    auto nThreads = threadpool.nThreads();

    vigra::HistogramOptions histogram_opt;
    histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal);

    // initialize local acc chain vectors in parallel
    auto localEdgeAccumulators = ThreadAccChainVectorType(nThreads);
    parallel::parallel_foreach(threadpool, nThreads,
    [&](int tid, int threadId) {
        auto & thisAcc = localEdgeAccumulators[threadId];
        thisAcc = AccChainVectorType(nLocal);
        if(accOptions.setMinMax){
            for(size_t edgeId; edgeId < nLocal; ++edgeId) {
                thisAcc[edgeId].setHistogramOptions(histogram_opt);
            }
        }
    });

    // initialize lifted acc chain vectors in parallel
    auto liftedEdgeAccumulators = ThreadAccChainVectorType(nThreads);
    parallel::parallel_foreach(threadpool, nThreads,
    [&](int tid, int threadId) {
        auto & thisAcc = liftedEdgeAccumulators[threadId];
        thisAcc = AccChainVectorType(nLifted);
        if(accOptions.setMinMax){
            for(size_t edgeId; edgeId < nLifted; ++edgeId) {
                thisAcc[edgeId].setHistogramOptions(histogram_opt);
            }
        }
    });

    const auto & offsets = lnh.offsets();
    const int pass = 1;

    // iterate over all affinity links and accumulate the associated
    // affinity edges
    tools::parallelForEachCoordinate(threadpool, affShape, [&](int tid, const Coord4 & affCoord) {

        Coord3 cU, cV;
        VigraCoord vc;
        const auto & offset = offsets[affCoord[0]];

        for(int d = 0; d < 3; ++d) {
            cU[d] = affCoord[d+1];
            cV[d] = affCoord[d+1] + offset[d];
            // range check
            if(cV[d] < 0 || cV[d] >= shape[d]) {
                return;
            }
        }

        const auto u = xtensor::read(labels, cU.asStdArray());
        const auto v = xtensor::read(labels, cV.asStdArray());

        // only do stuff if the labels are different
        if(u != v) {

            // we just update the vigra coord of label u
            for(int d = 0; d < 3; ++d) {
                vc = cU[d];
            }

            const auto val = xtensor::read(affinities, affCoord.asStdArray());
            auto e = rag.findEdge(u, v);
            if(e != -1) {
                auto & thisAccumulators = localEdgeAccumulators[tid];
                thisAccumulators[e].updatePassN(val, vc, pass);
            } else {
                auto & thisAccumulators = liftedEdgeAccumulators[tid];
                e = lnh.findEdge(u, v);
                thisAccumulators[e].updatePassN(val, vc, pass);
            }
        }

    });

    // merge the accumulators in parallel
    auto & localResultAccVec = localEdgeAccumulators.front();
    parallel::parallel_foreach(threadpool, localResultAccVec.size(),
    [&](const int tid, const int64_t edge){
        for(auto t=1; t<nThreads; ++t){
            localResultAccVec[edge].merge((localEdgeAccumulators[t])[edge]);
        }
    });
    f_local(localResultAccVec);

    auto & liftedResultAccVec = liftedEdgeAccumulators.front();
    parallel::parallel_foreach(threadpool, liftedResultAccVec.size(),
    [&](const int tid, const int64_t edge){
        for(auto t=1; t<nThreads; ++t){
            liftedResultAccVec[edge].merge((liftedEdgeAccumulators[t])[edge]);
        }
    });
    f_lifted(liftedResultAccVec);
}


// 10 features
template<class RAG, class LNH, class AFFINITIES, class FEATURE_ARRAY>
void accumulateLongRangeAffinities(
    const RAG & rag,
    const LNH & lnh,
    const AFFINITIES & affinities,
    const double minVal,
    const double maxVal,
    xt::xexpression<FEATURE_ARRAY> & localFeaturesExp,
    xt::xexpression<FEATURE_ARRAY> & liftedFeaturesExp,
    const int numberOfThreads = -1
){
    // TODO check that affinity channels and lnh axes and ranges agree
    // check that shapes off affs and labels agree
    namespace acc = vigra::acc;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    // TODO need to accumulate edge size here as well !!!
    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        // 1
        acc::Variance,    // 1
        Quantiles,        // 7
        acc::Count        // 1
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, float, SelectType> AccChainType;
    auto & localFeatures = localFeaturesExp.derived_cast();
    auto & liftedFeatures = liftedFeaturesExp.derived_cast();

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);
    const size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    auto accumulateLocal = [&](
        const std::vector<AccChainType> & edgeAccChainVec
    ){
        using namespace vigra::acc;

        parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
            const int tid, const int64_t edge
        ){
            const auto & chain = edgeAccChainVec[edge];
            const auto mean = get<acc::Mean>(chain);
            const auto quantiles = get<Quantiles>(chain);
            localFeatures(edge, 0) = replaceIfNotFinite(mean,     0.0);
            localFeatures(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
            for(auto qi=0; qi<7; ++qi)
                localFeatures(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
        });
    };

    auto accumulateLifted = [&](
        const std::vector<AccChainType> & edgeAccChainVec
    ){
        using namespace vigra::acc;

        parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
            const int tid, const int64_t edge
        ){
            const auto & chain = edgeAccChainVec[edge];
            const auto mean = get<acc::Mean>(chain);
            const auto quantiles = get<Quantiles>(chain);
            liftedFeatures(edge, 0) = replaceIfNotFinite(mean,     0.0);
            liftedFeatures(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
            for(auto qi=0; qi<7; ++qi)
                liftedFeatures(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
        });
    };

    accumulateLongRangeAffninitiesWithAccChain<AccChainType>(
        rag,
        lnh,
        affinities,
        threadpool,
        accumulateLocal,
        accumulateLifted,
        AccOptions(minVal, maxVal)
    );
}


}
}
