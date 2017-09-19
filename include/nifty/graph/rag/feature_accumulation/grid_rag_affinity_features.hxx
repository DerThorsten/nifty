#pragma once

#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/graph/rag/feature_accumulation/lifted_nh.hxx"

// accumulate features for flat superpixels with normal rag

namespace nifty{
namespace graph{


// TODO use block storage mechanism to make out of core
template<class EDGE_ACC_CHAIN, class RAG, class LNH, class AFFINITIES, class F_LOCAL, class F_LIFTED>
void accumulateLongRangeAffninitiesWithAccChain(
    const RAG & rag,
    const LNH & lnh,
    const AFFINITIES & affinities,
    // TODO parallelize properly
    //const parallel::ParallelOptions & pOpts,
    //parallel::ThreadPool & threadpool,
    F_LOCAL && f_local,
    F_LIFTED && f_lifted,
    const AccOptions & accOptions = AccOptions()
){
    


}


// 9 features
template<class RAG, class LNH, class AFFINITIES>
void accumulateLongRangeAffinities(
    const RAG & rag,
    const LNH & lnh,
    const AFFINITIES & affinities,
    const double minVal,
    const double maxVal,
    marray::View<float> & localFeatures,
    marray::View<float> & liftedFeatures,
    const int numberOfThreads = -1
){
    // TODO check that affinity channels and lnh axes and ranges
    // agree
    namespace acc = vigra::acc;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, float, SelectType> AccChainType;

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
        accumulateLocal,
        accumulateLifted,
        AccOptions(minVal, maxVal)
    );
}


}
}
