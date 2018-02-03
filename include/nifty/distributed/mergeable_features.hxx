#pragma once

#include "vigra/accumulator.hxx"
#include "nifty/distributed/distributed_graph.hxx"

namespace fs = boost::filesystem;
namespace acc = vigra::acc;

namespace nifty {
namespace distributed {


    ///
    // accumulator typedges
    ///
    typedef acc::UserRangeHistogram<40> FeatureHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<FeatureHistogram> Quantiles;

    typedef acc::Select<
        acc::Mean,        // 1
        acc::Variance,    // 1
        Quantiles,        // 7
        acc::Count        // 1
    > FeaturesType;
    //typedef acc::StandAloneAccumulatorChain<3, float, FeaturesType> AccumulatorChain;
    typedef acc::AccumulatorChain<float, FeaturesType> AccumulatorChain;

    typedef std::vector<AccumulatorChain> AccumulatorVector;
    typedef vigra::HistogramOptions HistogramOptions;

    typedef std::array<int, 3> OffsetType;


    template<class T,class U>
    inline T replaceIfNotFinite(const T & val, const U & replaceVal){
        if(std::isfinite(val))
            return val;
        else
            return replaceVal;
    }


    template<class FEATURE_ACCUMULATOR>
    inline void extractBlockFeaturesImpl(const std::string & groupPath,
                                         const std::string & blockPrefix,
                                         const std::string & dataPath,
                                         const std::string & dataKey,
                                         const std::string & labelPath,
                                         const std::string & labelKey,
                                         const std::vector<size_t> & blockIds,
                                         const std::string & tmpFeatureStorage,
                                         FEATURE_ACCUMULATOR && accumulator) {

        fs::path dataSetPath(dataPath);
        dataSetPath /= dataKey;
        auto data = z5::openDataset(dataSetPath.string());

        fs::path labelsSetPath(labelPath);
        labelsSetPath /= labelKey;
        auto labels = z5::openDataset(labelsSetPath.string());

        const std::vector<std::string> keys = {"roiBegin", "roiEnd"};

        std::vector<size_t> roiBegin(3);
        std::vector<size_t> roiEnd(3);

        // make path to the feature storage
        fs::path featureStorage(tmpFeatureStorage);
        fs::path blockStoragePath;

        fs::path blockPath;
        std::string blockKey;
        for(auto blockId : blockIds) {

            // get the path to the subgraph
            blockKey = blockPrefix + std::to_string(blockId);
            blockPath = fs::path(groupPath);
            blockPath /= blockKey;

            // load the graph
            Graph graph(blockPath.string());

            // continue if we don't have edges in this graph
            if(graph.numberOfEdges() == 0) {
                continue;
            }

            // load the bounding box
            z5::handle::Group group(blockPath.string());
            nlohmann::json j;
            z5::readAttributes(group, keys, j);

            const auto & jBegin = j[keys[0]];
            const auto & jEnd = j[keys[1]];

            for(unsigned axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = jBegin[axis];
                roiEnd[axis] = jEnd[axis];
            }

            // run the accumulator
            blockStoragePath = featureStorage;
            blockStoragePath /= "block_" + std::to_string(blockId);
            accumulator(graph, std::move(data), std::move(labels), roiBegin, roiEnd, blockStoragePath.string());
        }
    }


    ///
    // accumulator implementations
    ///


    // TODO need to accept path(s) for edge serialization
    // helper function to serialize edge features
    // unfortunately, we can't (trivially) serialize the state of the accumulators
    // so instead, we get the desired statistics and serialize those, to be merged
    // later heuristically
    inline void serializeDefaultEdgeFeatures(const AccumulatorVector & accumulators,
                                             const std::string & blockStoragePath) {
        // the number of features is hard-coded to 10 for now
        const std::vector<size_t> zero2Coord({0, 0});
        xt::xtensor<float, 2> values(Shape2Type({accumulators.size(), 10}));
        EdgeIndexType edgeId = 0;
        for(const auto & accumulator : accumulators) {
            // get the values from this accumulator
            const auto mean = replaceIfNotFinite(acc::get<acc::Mean>(accumulator), 0.0);
            values(edgeId, 0) = mean;
            values(edgeId, 1) = replaceIfNotFinite(acc::get<acc::Variance>(accumulator), 0.0);
            const auto & quantiles = acc::get<Quantiles>(accumulator);
            for(unsigned qi = 0; qi < 7; ++qi) {
                values(edgeId, 2 + qi) = replaceIfNotFinite(quantiles[qi], mean);
            }
            values(edgeId, 9) = replaceIfNotFinite(acc::get<acc::Count>(accumulator), 0.0);
            ++edgeId;
        }

        // serialize the features to z5 (TODO chunking / compression ?!)
        std::vector<size_t> shape = {accumulators.size(), 10};
        auto ds = z5::createDataset(blockStoragePath, "float32", shape, shape, false);
        z5::multiarray::writeSubarray<float>(ds, values, zero2Coord.begin());
    }


    // TODO arguments for serializaation
    // accumulate simple boundary map
    inline void accumulateBoundaryMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<size_t> & roiBegin,
                                      const std::vector<size_t> & roiEnd,
                                      const std::string & blockStoragePath,
                                      float dataMin, float dataMax) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<float, 3> DataArray;

        // get the shapes
        Shape3Type shape;
        CoordType blockShape;
        for(unsigned axis = 0; ++axis; axis < 3) {
            shape[axis] = roiEnd[axis] - roiBegin[axis];
            blockShape[axis] = shape[axis];
        }

        // load data and labels
        DataArray data(shape);
        LabelArray labels(shape);
        z5::multiarray::readSubarray<float>(dataDs, data, roiBegin.begin());
        z5::multiarray::readSubarray<NodeType>(labelsDs, labels, roiBegin.begin());

        // create nifty accumulator vector
        AccumulatorVector accumulators(graph.numberOfEdges());
        const int pass = 1;
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        // accumulate
        CoordType coord2;
        NodeType lU, lV;
        float fU, fV;
        EdgeIndexType edge;
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {
            lU = xtensor::read(labels, coord.asStdArray());
            for(size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);
                if(coord2[axis] < blockShape[axis]){
                    lV = xtensor::read(labels, coord2.asStdArray());
                    if(lU != lV){
                        edge = graph.findEdge(lU, lV);
                        fU = xtensor::read(data, coord.asStdArray());
                        fV = xtensor::read(data, coord.asStdArray());
                        accumulators[edge].updatePassN(fU, pass);
                        accumulators[edge].updatePassN(fV, pass);
                    }
                }
            }
        });

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, blockStoragePath);
    }


    // accumulate affinity maps
    inline void accumulateAffinityMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<size_t> & roiBegin,
                                      const std::vector<size_t> & roiEnd,
                                      const std::string & blockStoragePath,
                                      const std::vector<OffsetType> & offsets,
                                      const std::vector<size_t> & haloBegin,
                                      const std::vector<size_t> & haloEnd,
                                      float dataMin, float dataMax) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<float, 4> DataArray;
        typedef typename DataArray::shape_type Shape4Type;

        typedef nifty::array::StaticArray<int64_t, 4> AffCoordType;

        // get the shapes with halos
        std::vector<size_t> beginWithHalo(3), endWithHalo(3), affsBegin(4);
        const auto & volumeShape = labelsDs->shape();
        for(unsigned axis = 0; ++axis; axis < 3) {
            beginWithHalo[axis] = std::max(roiBegin[axis] - haloBegin[axis], 0UL);
            endWithHalo[axis] = std::min(roiEnd[axis] + haloEnd[axis], volumeShape[axis]);
            affsBegin[axis + 1] = beginWithHalo[axis];
        }
        // TODO if we want to be more general, we need to allow for non-zero channel offset here
        affsBegin[0] = 0;

        Shape3Type shape;
        Shape4Type affShape;
        CoordType blockShape;
        AffCoordType affBlockShape;
        for(unsigned axis = 0; ++axis; axis < 3) {
            shape[axis] = endWithHalo[axis] - beginWithHalo[axis];
            blockShape[axis] = shape[axis];
            affShape[axis + 1] = shape[axis];
            affBlockShape[axis + 1] = shape[axis];
        }
        affShape[0] = offsets.size();
        affBlockShape[0] = affShape[0];

        // load data and labels
        DataArray affs(affShape);
        LabelArray labels(shape);
        z5::multiarray::readSubarray<float>(dataDs, affs, affs.begin());
        z5::multiarray::readSubarray<NodeType>(labelsDs, labels, beginWithHalo.begin());

        // create nifty accumulator vector
        AccumulatorVector accumulators(graph.numberOfEdges());
        const int pass = 1;
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        // accumulate
        CoordType coord, coord2;
        NodeType lU, lV;
        EdgeIndexType edge;

        nifty::tools::forEachCoordinate(affBlockShape, [&](const AffCoordType & affCoord) {

            // the 0th affinitiy coordinate gives the channel index
            const auto & offset = offsets[affCoord[0]];
            for(unsigned axis = 0; axis < 3; ++axis) {
                coord[axis] = affCoord[axis + 1];
                coord2[axis] = affCoord[axis + 1] + offset[axis];

                // bounds check
                if(coord2[axis] < 0 || coord2[axis] > blockShape[axis]) {
                    return;
                }
            }

            lU = xtensor::read(labels, coord.asStdArray());
            lV = xtensor::read(labels, coord2.asStdArray());
            if(lU != lV){
                // for long range affinites, the uv pair may not be part of the region graph
                // so we need to check if the edge actually exists
                edge = graph.findEdge(lU, lV);
                if(edge != -1) {
                    accumulators[edge].updatePassN(xtensor::read(affs, affCoord.asStdArray()), pass);
                }
            }
        });

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, blockStoragePath);
    }


    ///
    // bind specific accumulators
    ///


    // TODO additional serialization arguments
    inline void extractBlockFeaturesFromBoundaryMaps(const std::string & groupPath,
                                                     const std::string & blockPrefix,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<size_t> & blockIds,
                                                     const std::string & tmpFeatureStorage,
                                                     float dataMin=0, float dataMax=1) {

        // TODO could also use the std::bind pattern and std::function
        // TODO capture additional arguments that we need for serialization
        auto accumulator = [dataMin, dataMax](
                const Graph & graph,
                std::unique_ptr<z5::Dataset> dataDs,
                std::unique_ptr<z5::Dataset> labelsDs,
                const std::vector<size_t> & roiBegin,
                const std::vector<size_t> & roiEnd,
                const std::string & blockStoragePath) {

            accumulateBoundaryMap(graph, std::move(dataDs), std::move(labelsDs),
                                  roiBegin, roiEnd, blockStoragePath,
                                  dataMin, dataMax);
        };

        extractBlockFeaturesImpl(groupPath, blockPrefix,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, tmpFeatureStorage,
                                 accumulator);
    }


    // TODO additional serialization arguments
    inline void extractBlockFeaturesFromAffinityMaps(const std::string & groupPath,
                                                     const std::string & blockPrefix,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<size_t> & blockIds,
                                                     const std::string & tmpFeatureStorage,
                                                     const std::vector<OffsetType> & offsets,
                                                     float dataMin=0, float dataMax=1) {
        // calculate max halos from the offsets
        std::vector<size_t> haloBegin(3), haloEnd(3);
        for(const auto & offset : offsets) {
            for(unsigned axis = 0; axis < 3; ++axis) {
                if(offset[axis] < 0) {
                    haloBegin[axis] = std::max(static_cast<size_t>(-offset[axis]), haloBegin[axis]);
                } else if(offset[axis] > 0) {
                    haloEnd[axis] = std::max(static_cast<size_t>(offset[axis]), haloEnd[axis]);
                }
            }
        }

        // TODO could also use the std::bind pattern and std::function
        // TODO capture additional arguments that we need for serialization
        auto accumulator = [dataMin, dataMax, &offsets, &haloBegin, &haloEnd](
                const Graph & graph,
                std::unique_ptr<z5::Dataset> dataDs,
                std::unique_ptr<z5::Dataset> labelsDs,
                const std::vector<size_t> & roiBegin,
                const std::vector<size_t> & roiEnd,
                const std::string & blockStoragePath) {

            accumulateAffinityMap(graph, std::move(dataDs), std::move(labelsDs),
                                  roiBegin, roiEnd, blockStoragePath,
                                  offsets, haloBegin, haloEnd,
                                  dataMin, dataMax);
        };

        extractBlockFeaturesImpl(groupPath, blockPrefix,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, tmpFeatureStorage,
                                 accumulator);
    }


    ///
    // Feature block merging
    ///


    inline void loadBlockFeatures(const std::string & blockFeaturePath,
                                  xt::xtensor<float, 2> & features) {
        const std::vector<size_t> zero2Coord({0, 0});
        auto featDs = z5::openDataset(blockFeaturePath);
        z5::multiarray::writeSubarray<float>(featDs, features, zero2Coord.begin());
    }


    inline void mergeFeaturesForSingleEdge(xt::xtensor<float, 2> & tmpFeatures, xt::xtensor<float, 2> & targetFeatures,
                                           const EdgeIndexType tmpId, const EdgeIndexType targetId,
                                           std::vector<bool> & hasFeatures) {
        if(hasFeatures[targetId]) {

            // index 9 is the count
            auto nSamplesA = targetFeatures(targetId, 9);
            auto nSamplesB = tmpFeatures(tmpId, 9);
            auto nSamplesTot = nSamplesA + nSamplesB;
            auto ratioA = nSamplesA / nSamplesTot;
            auto ratioB = nSamplesB / nSamplesTot;

            // merge the mean
            float meanA = targetFeatures(targetId, 0);
            float meanB = tmpFeatures(tmpId, 0);
            float newMean = ratioA * meanA + ratioB * meanB;
            targetFeatures(targetId, 0) = newMean;

            // merge the variance (feature id 1)
            // see https://stackoverflow.com/questions/1480626/merging-two-statistical-result-sets
            float varA = targetFeatures(targetId, 1);
            float varB = tmpFeatures(tmpId, 1);
            targetFeatures(targetId, 1) = ratioA * (varA + (meanA - newMean) * (meanA - newMean));
            targetFeatures(targetId, 1) += ratioB * (varB + (meanB - newMean) * (meanB - newMean));

            // merge the min (feature id 2)
            targetFeatures(targetId, 2) = std::min(targetFeatures(targetId, 2), tmpFeatures(tmpId, 2));

            // merge the quantiles (not min and max !) via weighted average
            // this is not correct, but the best we can do for now
            for(size_t featId = 3; featId < 8; ++featId) {
                targetFeatures(targetId, featId) =  ratioA * targetFeatures(targetId, featId) + ratioB * tmpFeatures(tmpId, featId);
            }

            // merge the max (feature id 8)
            targetFeatures(targetId, 8) = std::max(targetFeatures(targetId, 8), tmpFeatures(tmpId, 8));
            // merge the count (feature id 9)
            targetFeatures(targetId, 9) = nSamplesTot;

        } else {

            for(size_t featId = 0; featId < 10; ++featId) {
                targetFeatures(targetId, featId) = tmpFeatures(tmpId, featId);
            }
            hasFeatures[targetId] = true;
        }
    }


    inline void mergeEdgeFeaturesForBlocks(const std::string & graphBlockPrefix,
                                           const std::string & featureBlockPrefix,
                                           const size_t edgeIdBegin,
                                           const size_t edgeIdEnd,
                                           const std::vector<size_t> & blockIds,
                                           nifty::parallel::ThreadPool & threadpool,
                                           const std::string & featuresOut) {
        //
        const size_t nEdges = edgeIdEnd - edgeIdBegin;
        Shape2Type fShape = {nEdges, 10};
        Shape2Type tmpInit = {1, 10};

        // initialize per thread data
        const size_t nThreads = threadpool.nThreads();
        struct PerThreadData {
            xt::xtensor<float, 2> features;
            xt::xtensor<float, 2> tmpFeatures;
            std::vector<bool> edgeHasFeatures;
        };
        std::vector<PerThreadData> perThreadDataVector(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&](const int t, const int tId){
            auto & ptd = perThreadDataVector[tId];
            ptd.features = xt::xtensor<float, 2>(fShape);
            ptd.tmpFeatures = xt::xtensor<float, 2>(tmpInit);
            ptd.edgeHasFeatures = std::vector<bool>(nEdges, false);
        });

        // iterate over the block ids
        const size_t nBlocks = blockIds.size();
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int tId, const int blockIndex){

            auto blockId = blockIds[blockIndex];
            auto & perThreadData = perThreadDataVector[tId];
            auto & features = perThreadData.features;
            auto & hasFeatures = perThreadData.edgeHasFeatures;
            auto & blockFeatures = perThreadData.tmpFeatures;

            // load edge ids for the block
            const std::string blockGraphPath = graphBlockPrefix + std::to_string(blockId);
            std::vector<EdgeIndexType> blockEdgeIndices;
            loadEdgeIndices(blockGraphPath, blockEdgeIndices);
            const size_t nEdgesBlock = blockEdgeIndices.size();
            std::unordered_map<EdgeIndexType, EdgeIndexType> toDenseBlockId;
            size_t denseBlockId = 0;
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                toDenseBlockId[edgeId] = denseBlockId;
                ++denseBlockId;
            }

            // generate mapping of global id to dense id in the block

            // load features for the block
            // first resize the tmp features, if necessary
            const std::string blockFeaturePath = featureBlockPrefix + std::to_string(blockId);
            if(blockFeatures.shape()[0] != nEdgesBlock) {
                blockFeatures.reshape({nEdgesBlock, 10});
            }
            loadBlockFeatures(blockFeaturePath, blockFeatures);

            // iterate over the edges in this block and merge edge features if they are in our edge range
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                // only merge if we are in the edge range
                if(edgeId >= edgeIdBegin && edgeId < edgeIdEnd) {
                    // find the corresponding ids in the dense block edges
                    // and in the dense edge range
                    auto rangeEdgeId = edgeId - edgeIdBegin;
                    auto blockEdgeId = toDenseBlockId[edgeId];
                    mergeFeaturesForSingleEdge(blockFeatures, features, blockEdgeId, rangeEdgeId, hasFeatures);
                }
            }

        });

        // merge features for each edge
        auto & features = perThreadDataVector[0].features;
        auto & hasFeatureVector = perThreadDataVector[0].edgeHasFeatures;
        nifty::parallel::parallel_foreach(threadpool, nEdges, [&](const int tId, const EdgeIndexType edgeId){
            for(int threadId = 1; threadId < nThreads; ++threadId) {
                auto & perThreadData = perThreadDataVector[threadId];
                if(perThreadData.edgeHasFeatures[edgeId]) {
                    mergeFeaturesForSingleEdge(features, perThreadData.features, edgeId, edgeId, hasFeatureVector);
                }
            }
        });

        // TODO we coudl parallelize this over the out chunks
        // serialize the edge features
        auto dsOut = z5::openDataset(featuresOut);
        const std::vector<size_t> zero2Coord({0, 0});
        z5::multiarray::writeSubarray<float>(dsOut, features, zero2Coord.begin());
    }


    inline void findRelevantBlocks(const std::string & graphBlockPrefix,
                                   const size_t numberOfBlocks,
                                   const size_t edgeIdBegin,
                                   const size_t edgeIdEnd,
                                   nifty::parallel::ThreadPool & threadpool,
                                   std::vector<size_t> & blockIds) {

        const size_t nThreads = threadpool.nThreads();
        std::vector<std::set<size_t>> perThreadData(nThreads);

        nifty::parallel::parallel_foreach(threadpool, numberOfBlocks, [&](const int tId, const int blockId) {

            const std::string blockPath = graphBlockPrefix + std::to_string(blockId);
            std::vector<EdgeIndexType> blockEdgeIndices;
            bool haveEdges = loadEdgeIndices(blockPath, blockEdgeIndices);
            if(!haveEdges) {
                return;
            }

            // first we check, if the block has overlap with at least one of our edges
            // (and thus if it is relevant) by a simple range check

            auto minmax = std::minmax_element(blockEdgeIndices.begin(), blockEdgeIndices.end());
            EdgeIndexType blockMinEdge = *minmax.first;
            EdgeIndexType blockMaxEdge = *minmax.second;

            // TODO these things might be off by one index ...
            // range check
            bool rangeBeginIsBiggerBlock  = (edgeIdBegin > blockMinEdge) && (edgeIdBegin >= blockMaxEdge);
            bool rangeBeginIsSmallerBlock = (edgeIdBegin < blockMinEdge) && (edgeIdBegin <= blockMaxEdge);

            bool rangeEndIsBiggerBlock  = (edgeIdEnd > blockMinEdge) && (edgeIdEnd >= blockMaxEdge);
            bool rangeEndIsSmallerBlock = (edgeIdEnd < blockMinEdge) && (edgeIdEnd <= blockMaxEdge);

            if((rangeBeginIsBiggerBlock && rangeEndIsBiggerBlock) || ((rangeBeginIsSmallerBlock && rangeEndIsSmallerBlock))) {
                return;
            }

            perThreadData[tId].insert(blockId);
        });

        // merge the relevant blocks
        auto & blockIdsSet = perThreadData[0];
        for(int tId = 1; tId < nThreads; ++tId) {
            blockIdsSet.insert(perThreadData[tId].begin(), perThreadData[tId].end());
        }

        // write to the out vector
        blockIds.resize(blockIdsSet.size());
        std::copy(blockIdsSet.begin(), blockIdsSet.end(), blockIds.begin());
    }


    inline void mergeFeatureBlocks(const std::string & graphBlockPrefix,
                                   const std::string & featureBlockPrefix,
                                   const std::string & featuresOut,
                                   const size_t numberOfBlocks,
                                   const size_t edgeIdBegin,
                                   const size_t edgeIdEnd,
                                   const int numberOfThreads=1) {
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);

        // find the relevant blocks, that have overlap with our edge ids
        std::vector<size_t> blockIds;
        findRelevantBlocks(graphBlockPrefix, numberOfBlocks,
                           edgeIdBegin, edgeIdEnd, threadpool,
                           blockIds);

        // merge all edges in the edge range
        mergeEdgeFeaturesForBlocks(graphBlockPrefix, featureBlockPrefix,
                                   edgeIdBegin, edgeIdEnd,
                                   blockIds,
                                   threadpool, featuresOut);
    }


}
}
