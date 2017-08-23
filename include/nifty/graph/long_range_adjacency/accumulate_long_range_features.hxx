#pragma once

#include "nifty/parallel/threadpool.hxx"

#include "nifty/tools/array_tools.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"


namespace nifty {
namespace graph {


// TODO
// get the long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
template<class RAG>
void getLongRangeAdjacency(
    const RAG & rag,
    const size_t longRange,
    std::vector<std::pair<typename RAG::LabelType, typename RAG::LabelType>> & adjacencyOut,
    const int numberOfThreads=-1
) {
    typedef array::StaticArray<int64_t,3> Coord;
    typedef array::StaticArray<int64_t,2> Coord2;
    typedef typename RAG::LabelsProxy LabelsProxy;
    typedef typename LabelsProxy::BlockStorageType LabelStorage;

    typedef typename RAG::LabelType LabelType;
    typedef std::set<std::pair<LabelType, LabelType>> AdjacencySet;

    // instantiate threadpool and get the actual number of threads
    parallel::ThreadPool threadpool(numberOfThreads);
    auto nThreads = threadpool.nThreads();

    // instantiate thread data (= adjacency set for each thread)
    std::vector<AdjacencySet> threadData(nThreads);

    // labels proxy and shape
    const auto & labelsProxy = rag.labelsProxy();
    const auto & shape = labelsProxy.shape();

    // instantiate the slice shapes
    const Coord2 sliceShape2({shape[1], shape[2]});
    const Coord  sliceShape3({1L, shape[1], shape[2]});

    // instantiate the label storage
    LabelStorage labelsAStorage(threadpool, sliceShape3, nThreads);
    LabelStorage labelsBStorage(threadpool, sliceShape3, nThreads);

    // we don't need to take into account the 2 uppermost slices, because they don't have any 
    // long tange neighbors
    size_t nSlices = shape[0] - 2;
    // iterate over all the slices z and find the adjacency to the slices above,
    // from z+2 to z+longRange
    parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int slice) {

        // get thread
        auto & threadAdjacency = threadData[tid];

        // get lower segmentation
        Coord beginA ({int64_t(slice), 0L, 0L});
        Coord endA({int64_t(slice + 1), shape[1], shape[2]});
        auto labelsA = labelsAStorage.getView(tid);
        labelsProxy.readSubarray(beginA, endA, labelsA);
        auto labelsASqueezed = labelsA.squeezedView();

        // get view for upper segmentation
        auto labelsB = labelsBStorage.getView(tid);

        for(int64_t z = 2; z <= longRange; ++z) {

            // we continue if the long range affinity would reach out of the data
            if(slice + z >= shape[0]) {
                continue;
            }

            // get upper segmentation
            Coord beginB ({slice + z, 0L, 0L});
            Coord endB({slice + z + 1, shape[1], shape[2]});
            labelsProxy.readSubarray(beginB, endB, labelsB);
            auto labelsBSqueezed = labelsB.squeezedView();

            // iterate over the xy-coordinates
            LabelType lU, lV;
            tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
                lU = labelsASqueezed(coord.asStdArray());
                lV = labelsASqueezed(coord.asStdArray());
                threadAdjacency.insert(std::make_pair(std::min(lU, lV), std::max(lU, lV)));
            });
        }
    });

    // write the results to out vector
    // out size and thread offsets
    size_t totalSize = 0;
    std::vector<size_t> threadOffsets(nThreads);
    for(int tId = 0; tId < nThreads; ++tId) {
        auto threadSize = threadData[tId].size();
        threadOffsets[tId] = totalSize;
        totalSize += threadSize;
    }
    adjacencyOut.resize(totalSize);
    parallel::parallel_foreach(threadpool, nThreads, [&](const int tId, const int threadId){
        const auto & threadAdjacency = threadData[threadId];
        std::copy(threadAdjacency.begin(), threadAdjacency.end(), adjacencyOut.begin() + threadOffsets[threadId]);
    });
}


// accumulate features for long range adjacency along the z (anisotropic) axis
// assumes flat superpixels !
template<class EDGE_ACC_CHAIN, class RAG, class AFFINITIES, class F>
void accumulateLongRangeFeaturesWithAccChain(
    const RAG & rag,
    const AFFINITIES & affinities,
    const std::vector<std::pair<typename RAG::LabelType, typename RAG::LabelType>> & adjacency,
    const size_t longRange,
    parallel::ThreadPool & threadpool,
    F && f,
    const AccOptions & accOptions = AccOptions()
) {
    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::LabelType LabelType;
    typedef typename DATA::DataType DataType;

    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType> AccChainVectorType;

    const size_t actualNumberOfThreads = threadpool.nThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();

    size_t nSlices = shape[0] - 2;
    size_t nEdges = adjacency.size();

    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1L, shape[1], shape[2]});

    // edge acc vectors for multiple threads
    std::vector<AccChainVectorType> perThreadAccChainVector(actualNumberOfThreads);
    parallel::parallel_foreach(threadpool, actualNumberOfThreads,
    [&](const int tid, const int64_t i){
        perThreadAccChainVector[i] = AccChainVectorType(nEdges);
    });

    // set the accumulator chain options
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

    const int pass = 1
    {
        // label and data storages
        LabelBlockStorage  labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage  labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        DataBlockStorage   dataAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        DataBlockStorage   dataBStorage(threadpool, sliceShape3, actualNumberOfThreads);

        parallel::parallel_foreach(threadpool, nSlices, [&](const int tid, const int64_t slice){

            auto & threadAccChainVec = perThreadAccChainVector[tid];

            Coord beginA ({slice, 0L, 0L});
            Coord endA({slice + 1, shape[1], shape[2]});

            auto labelsA = labelsAStorage.getView(tid);
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            auto dataA = dataAStorage.getView(tid);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = dataA.squeezedView();

            // process upper slice
            Coord beginB = Coord({sliceIdB,   0L,       0L});
            Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});
            marray::View<LabelType> labelsBSqueezed;

            // read labels and data for upper slice
            auto labelsB = labelsBStorage.getView(tid);
            labelsProxy.readSubarray(beginB, endB, labelsB);
            labelsBSqueezed = labelsB.squeezedView();
            auto dataB = dataBStorage.getView(tid);
            tools::readSubarray(data, beginB, endB, dataB);
            auto dataBSqueezed = dataB.squeezedView();
        
            for(int64_t z = 2; z <= longRange; ++z) {

                // we continue if the long range affinity would reach out of the data
                if(slice + z >= shape[0]) {
                    continue;
                }

                // TODO TODO TODO
                // accumulate the long range features
                accumulateLongRangeFeaturesForSlice(
                    threadAccChainVec,
                    sliceShape2,
                    labelsASqueezed,
                    labelsBSqueezed,
                    adjacency,
                    dataASqueezed,
                    dataBSqueezed,
                    pass,
                    slice,
                    slice+z+,
                    accOptions.zDirection
                );
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


template<class RAG, class AFFINITIES, class OUTPUT>
void accumulateLongRangeFeatures(
    const RAG & rag,
    const AFFINITIES & affinities,
    const std::vector<std::pair<typename RAG::LabelType, typename RAG::LabelType>> & adjacency,
    const size_t longRange,
    OUTPUT & featuresOut,
    const double minVal,
    const double maxVal,
    const int zDirection = 0,
    const int numberOfThreads = -1
) {

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

    // accumulator function
    auto accFunction = [&threadpool, &featuresOut](
        const std::vector<AccChainType> & edgeAccChainVec
    ){
        using namespace vigra::acc;
        typedef array::StaticArray<int64_t, 2> FeatCoord;

        parallel::parallel_foreach(threadpool, edgeAccChainVec.size(),[&](
            const int tid, const int64_t edge
        ){
            const auto & chain = edgeAccChainVec[edge];
            const auto mean = get<acc::Mean>(chain);
            const auto quantiles = get<Quantiles>(chain);
            edgeFeaturesOut(edge, 0) = replaceIfNotFinite(mean, 0.0);
            edgeFeaturesOut(edge, 1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
            for(auto qi=0; qi<7; ++qi)
                edgeFeaturesOut(edge, 2+qi) = replaceIfNotFinite(quantiles[qi], mean);
        });

    };

    accumulateLongRangeFeaturesWithAccChain<AccChainType>(
        rag,
        affinities,
        adjacency,
        longRange,
        threadpool,
        zDirection,
        accFunction,
        AccOptions(minVal, maxVal, zDirection)
    );

}


} // end namespace graph
} // end namespace nifty
