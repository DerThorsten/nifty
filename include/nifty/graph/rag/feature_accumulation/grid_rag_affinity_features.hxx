#pragma once

#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/graph/rag/feature_accumulation/lifted_nh.hxx"

// accumulate features for flat superpixels with normal rag

namespace nifty{
namespace graph{


// TODO parallelize properly
// TODO use block storage mechanism to make out of core
template<class EDGE_ACC_CHAIN, class RAG, class LNH, class AFFINITIES, class F_LOCAL, class F_LIFTED>
void accumulateLongRangeAffninitiesWithAccChain(
    const RAG & rag,
    const LNH & lnh,
    const AFFINITIES & affinities,
    parallel::ThreadPool & threadpool,
    F_LOCAL && f_local,
    F_LIFTED && f_lifted,
    const AccOptions & accOptions = AccOptions()
){
    typedef array::StaticArray<int64_t, 3> Coord3;
    typedef array::StaticArray<int64_t, 4> Coord4;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;

    const auto & labels = rag.labelsProxy().labels();

    Coord3 shape;
    Coord4 affShape;
    affShape[0] = affinities.shape(0);
    for(int d = 0; d < 3; ++d) {
        shape[d] = labels.shape(d);
        affShape[d] = affinities.shape(d+1);
    }

    // only single threaded for now
    // accumulator chain vectors for local and lifted edges
    size_t nLocal = rag.edgeIdUpperBound() + 1;
    size_t nLifted = lnh.edgeIdUpperBound() + 1;
    AccChainVectorType localEdgeAccumulators(nLocal);
    AccChainVectorType liftedEdgeAccumulators(nLifted);

    // set the histogram options
    if(accOptions.setMinMax){
        vigra::HistogramOptions histogram_opt;
        histogram_opt = histogram_opt.setMinMax(accOptions.minVal, accOptions.maxVal);
        // for local accumumlators
        parallel::parallel_foreach(threadpool, nLocal,
        [&](int tid, int edgeId){
            localEdgeAccumulators[edgeId].setHistogramOptions(histogram_opt);
        });
        // for lifted accumumlators
        parallel::parallel_foreach(threadpool, nLifted,
        [&](int tid, int edgeId){
            liftedEdgeAccumulators[edgeId].setHistogramOptions(histogram_opt);
        });
    }

    // axes and reanges from the lifted nhood
    const auto & axes = lnh.axes();
    const auto & ranges = lnh.ranges();

    Coord4 affCoord;
    Coord3 cU, cV;
    VigraCoord vc;
    int axis, range;
    int pass = 1;

    size_t nLinks = affinities.size();
    // iterate over all affinity links and accumulate the associated
    // affinity edges
    for(size_t linkId = 0; linkId < nLinks; ++linkId) {

        affinities.indexToCoordinates(linkId, affCoord.begin());
        axis  = axes[affCoord[0]];
        range = ranges[affCoord[0]];

        for(int d = 0; d < 3; ++d) {
            cU[d] = affCoord[d+1];
            cV[d] = affCoord[d+1];
        }
        cV[axis] += range;
        // range check
        if(cV[axis] >= shape[axis] || cV[axis] < 0) {
            continue;
        }
        auto u = labels(cU.asStdArray());
        auto v = labels(cV.asStdArray());

        // only do stuff if the labels are different
        if(u != v) {

            // we just update the vigra coord of label u
            for(int d = 0; d < 3; ++d) {
                vc = cU[d];
            }

            auto val = affinities(affCoord.asStdArray());
            auto e = rag.findEdge(u, v);
            if(e != -1) {
                localEdgeAccumulators[e].updatePassN(val, vc, pass);
            } else {
                // asser that this really exists ?!
                e = lnh.findEdge(u, v);
                liftedEdgeAccumulators[e].updatePassN(val, vc, pass);
            }
        }
    }
    f_local(localEdgeAccumulators);
    f_lifted(liftedEdgeAccumulators);
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
    // TODO check that affinity channels and lnh axes and ranges agree
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
        threadpool,
        accumulateLocal,
        accumulateLifted,
        AccOptions(minVal, maxVal)
    );
}


}
}
