#pragma once

#include "nifty/parallel/threadpool.hxx"

#include "nifty/tools/array_tools.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/graph/long_range_adjacency/long_range_adjacency.hxx"


namespace nifty {
namespace graph {

template<class ADJACENCY, class ACC_CHAIN_VECTOR, class COORD,
         class LABELS, class AFFS>
void accumulateLongRangeFeaturesForSlice(
    const ADJACENCY & adj,
    ACC_CHAIN_VECTOR & accChainVec,
    const COORD & sliceShape2,
    const xt::xexpression<LABELS> & labelsAExp,
    const xt::xexpression<LABELS> & labelsBExp,
    const xt::xexpression<AFFS> & affinitiesExp,
    const int pass,
    const size_t slice,
    const size_t targetSlice,
    const size_t edgeOffset
) {
    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    typedef typename ADJACENCY::LabelType LabelType;

    const auto & labelsA = labelsAExp.derived_cast();
    const auto & labelsB = labelsBExp.derived_cast();
    const auto & affinities = affinitiesExp.derived_cast();

    VigraCoord vigraCoord;
    LabelType lU, lV;
    float aff;
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){

        // labels are different for different slices by default!
        lU = xtensor::read(labelsA, coord.asStdArray());
        lV = xtensor::read(labelsB, coord.asStdArray());
        // affinity
        aff = xtensor::read(affinities, coord.asStdArray());

        vigraCoord[0] = slice;
        for(int d = 1; d < 3; ++d){
            vigraCoord[d] = coord[d-1];
        }

        if(adj.hasIgnoreLabel() && (lU == 0 || lV == 0)) {
            return;
        }

        const auto edge = adj.findEdge(lU, lV) - edgeOffset;
        accChainVec[edge].updatePassN(aff, vigraCoord, pass);
    });

}


// accumulate features for long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
template<class EDGE_ACC_CHAIN, class ADJACENCY, class LABELS, class AFFINITIES, class F>
void accumulateLongRangeFeaturesWithAccChain(
    const ADJACENCY & adj,
    const LABELS & labels,
    const AFFINITIES & affinities,
    parallel::ThreadPool & threadpool,
    F && f,
    const int zDirection
) {
    typedef typename AFFINITIES::DataType DataType;
    typedef typename LABELS::DataType LabelType;

    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<LabelType> LabelBlockStorage;

    typedef array::StaticArray<int64_t, 4> Coord4;
    typedef array::StaticArray<int64_t, 3> Coord3;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType> EdgeAccChainVectorType;

    const size_t actualNumberOfThreads = threadpool.nThreads();

    const auto & shape = adj.shape();

    const size_t nSlices = shape[0] - 2;
    const size_t nEdges = adj.numberOfEdges();

    // convention for zDirection: 1 -> affinties go to upper slices,
    // 2 -> affinities go to lower slices
    const bool affsToUpper = zDirection == 1;

    Coord2 sliceShape2({shape[1], shape[2]});
    Coord3 sliceShape3({1L, shape[1], shape[2]});
    Coord4 sliceShape4({1L, 1L, shape[1], shape[2]});

    const int pass = 1;
    {
        // label and data storages
        LabelBlockStorage labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        DataBlockStorage affinityStorage(threadpool, sliceShape4, actualNumberOfThreads);

        // affinities are in range 0 to 1 -> we hardcode this !!
        vigra::HistogramOptions histoOptions;
        histoOptions.setMinMax(0., 1.);

        parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int64_t slice){

            // init this accumulatore chain
            EdgeAccChainVectorType accChainVec(adj.numberOfEdgesInSlice(slice));
            // set minmax for accumulator chains
            for(size_t edge = 0; edge < accChainVec.size(); ++edge){
                accChainVec[edge].setHistogramOptions(histoOptions);
            }
            const size_t edgeOffset = adj.edgeOffset(slice);

            Coord3 beginA({slice, 0L, 0L});
            Coord3 endA({slice + 1, shape[1], shape[2]});

            auto labelsA = labelsAStorage.getView(tid);
            tools::readSubarray(labels, beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            // initialize the affinity storage and coordinates
            auto affs = affinityStorage.getView(tid);
            Coord4 beginAff({0L, 0L, 0L, 0L});
            Coord4 endAff({0L, 0L, shape[1], shape[2]});

            // init view for labelsB
            auto labelsB = labelsBStorage.getView(tid);

            int64_t targetSlice;
            size_t channel;
            for(int64_t z = 2; z <= adj.range(); ++z) {

                targetSlice = slice + z;
                // we break if the long range affinity would reach out of the data
                if(targetSlice >= shape[0]) {
                    break;
                }

                // get the correct affinity channel
                channel = z - 2;
                beginAff[0] = channel;
                endAff[0] = channel+1;

                // if affinities point to upper slices, we read the affinity channel from the
                // lower slice
                if(affsToUpper) {
                    beginAff[1] = slice;
                    endAff[1] = slice + 1;
                }
                // otherwise we read them from the upper slice
                else {
                    beginAff[1] = targetSlice;
                    endAff[1] = targetSlice + 1;
                }

                // read and squeeze affinities
                tools::readSubarray(affinities, beginAff, endAff, affs);
                auto affsSqueezed = affs.squeezedView();

                // read upper labels
                Coord3 beginB({targetSlice,   0L,       0L});
                Coord3 endB({targetSlice + 1, shape[1], shape[2]});
                tools::readSubarray(labels, beginB, endB, labelsB);
                auto labelsBSqueezed = labelsB.squeezedView();

                // accumulate the long range features
                accumulateLongRangeFeaturesForSlice(
                    adj,
                    accChainVec,
                    sliceShape2,
                    labelsASqueezed,
                    labelsBSqueezed,
                    affsSqueezed,
                    pass,
                    slice,
                    targetSlice,
                    edgeOffset
                );
            }

            // callback to write out features
            f(accChainVec, edgeOffset);
        });
    }
}


template<class ADJACENCY, class LABELS, class AFFINITIES, class OUTPUT>
void accumulateLongRangeFeatures(
    const ADJACENCY & adj,
    const LABELS & labels,
    const AFFINITIES & affinities,
    OUTPUT & featuresOut,
    const int zDirection,
    const int numberOfThreads = -1
) {

    namespace acc = vigra::acc;
    typedef float DataType;

    typedef acc::UserRangeHistogram<40> SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram> Quantiles;

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

    // accumulator function
    auto accFunction = [&threadpool, &featuresOut](
        const std::vector<AccChainType> & edgeAccChainVec,
        const size_t edgeOffset
    ){
        using namespace vigra::acc;
        typedef array::StaticArray<int64_t, 2> FeatCoord;

        const uint64_t nEdges = edgeAccChainVec.size();
        const uint64_t nStats = 9;

        xt::xtenspor<float, 2> featuresTemp({nEdges, nStats});

        for(auto edge = 0; edge < nEdges; ++edge) {
            const auto & chain = edgeAccChainVec[edge];
            const auto mean = get<acc::Mean>(chain);
            const auto quantiles = get<Quantiles>(chain);
            featuresTemp(edge, 0) = replaceIfNotFinite(mean, 0.0);
            featuresTemp(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
            for(auto qi=0; qi<7; ++qi)
                featuresTemp(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
        }

        FeatCoord begin({int64_t(edgeOffset),0L});
        FeatCoord end({edgeOffset+nEdges, nStats});

        tools::writeSubarray(featuresOut, begin, end, featuresTemp);
    };

    accumulateLongRangeFeaturesWithAccChain<AccChainType>(
        adj,
        labels,
        affinities,
        threadpool,
        accFunction,
        zDirection
    );

}


} // end namespace graph
} // end namespace nifty
