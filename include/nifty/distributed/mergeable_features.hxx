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
    > FeatureSelection;

    typedef double FeatureType;
    typedef acc::AccumulatorChain<FeatureType, FeatureSelection> AccumulatorChain;

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
    inline void extractBlockFeaturesImpl(const std::string & blockPrefix,
                                         const std::string & dataPath,
                                         const std::string & dataKey,
                                         const std::string & labelPath,
                                         const std::string & labelKey,
                                         const std::vector<size_t> & blockIds,
                                         const std::string & tmpFeatureStorage,
                                         FEATURE_ACCUMULATOR && accumulator) {

        fs::path dataSetPath(dataPath);
        dataSetPath /= dataKey;

        fs::path labelsSetPath(labelPath);
        labelsSetPath /= labelKey;

        const std::vector<std::string> keys = {"roiBegin", "roiEnd"};

        // make path to the feature storage
        fs::path featureStorage(tmpFeatureStorage);
        fs::path blockStoragePath;

        for(auto blockId : blockIds) {

            // we move the unique ptr, so after the loop it will be null
            // hence we need to create the unique-ptr in each loop again.
            // TODO the proper way to do this would be with shared_ptrs,
            // but for this some of the API needs to change
            auto data = z5::openDataset(dataSetPath.string());
            auto labels = z5::openDataset(labelsSetPath.string());

            // get the path to the subgraph
            const std::string blockPath = blockPrefix + std::to_string(blockId);

            // load the graph
            Graph graph(blockPath);

            // continue if we don't have edges in this graph
            if(graph.numberOfEdges() == 0) {
                continue;
            }

            // load the bounding box
            z5::handle::Group group(blockPath);
            nlohmann::json j;
            z5::readAttributes(group, keys, j);

            const auto & jBegin = j[keys[0]];
            const auto & jEnd = j[keys[1]];

            std::vector<size_t> roiBegin(3);
            std::vector<size_t> roiEnd(3);
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
        xt::xtensor<FeatureType, 2> values(Shape2Type({accumulators.size(), 10}));
        EdgeIndexType edgeId = 0;

        for(const auto & accumulator : accumulators) {
            // get the values from this accumulator
            const FeatureType mean = replaceIfNotFinite(acc::get<acc::Mean>(accumulator), 0.0);

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
        auto ds = z5::createDataset(blockStoragePath, "float64", shape, shape, false);
        z5::multiarray::writeSubarray<FeatureType>(ds, values, zero2Coord.begin());
    }


    // accumulate simple boundary map
    template<class InputType>
    inline void accumulateBoundaryMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<size_t> & roiBegin,
                                      const std::vector<size_t> & roiEnd,
                                      const std::string & blockStoragePath,
                                      FeatureType dataMin, FeatureType dataMax) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<InputType, 3> DataArray;

        // get the shapes
        Shape3Type shape;
        CoordType blockShape;
        for(unsigned axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - roiBegin[axis];
            blockShape[axis] = shape[axis];
        }

        // load data and labels
        DataArray data(shape);
        LabelArray labels(shape);
        z5::multiarray::readSubarray<InputType>(dataDs, data, roiBegin.begin());
        z5::multiarray::readSubarray<NodeType>(labelsDs, labels, roiBegin.begin());

        // create nifty accumulator vector
        AccumulatorVector accumulators(graph.numberOfEdges());
        const int pass = 1;
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        const bool byteInput = typeid(InputType) == typeid(uint8_t);

        // accumulate
        if(byteInput) {
            nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {
                const NodeType lU = xtensor::read(labels, coord.asStdArray());
                CoordType coord2;
                for(size_t axis = 0; axis < 3; ++axis){
                    makeCoord2(coord, coord2, axis);
                    if(coord2[axis] < blockShape[axis]){
                        const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                        if(lU != lV){
                            const EdgeIndexType edge = graph.findEdge(lU, lV);
                            const auto fU = xtensor::read(data, coord.asStdArray());
                            const auto fV = xtensor::read(data, coord2.asStdArray());
                            FeatureType fUf = static_cast<FeatureType>(fU);
                            FeatureType fVf = static_cast<FeatureType>(fV);
                            accumulators[edge].updatePassN(fUf / 255., pass);
                            accumulators[edge].updatePassN(fVf / 255., pass);
                        }
                    }
                }
            });
        } else {
            nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {
                const NodeType lU = xtensor::read(labels, coord.asStdArray());
                CoordType coord2;
                for(size_t axis = 0; axis < 3; ++axis){
                    makeCoord2(coord, coord2, axis);
                    if(coord2[axis] < blockShape[axis]){
                        const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                        if(lU != lV){
                            const EdgeIndexType edge = graph.findEdge(lU, lV);
                            const auto fU = xtensor::read(data, coord.asStdArray());
                            const auto fV = xtensor::read(data, coord2.asStdArray());
                            accumulators[edge].updatePassN(fU, pass);
                            accumulators[edge].updatePassN(fV, pass);
                        }
                    }
                }
            });
        }

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, blockStoragePath);
    }


    // accumulate affinity maps
    template<class InputType>
    inline void accumulateAffinityMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<size_t> & roiBegin,
                                      const std::vector<size_t> & roiEnd,
                                      const std::string & blockStoragePath,
                                      const std::vector<OffsetType> & offsets,
                                      const std::vector<size_t> & haloBegin,
                                      const std::vector<size_t> & haloEnd,
                                      FeatureType dataMin, FeatureType dataMax) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<InputType, 4> DataArray;
        typedef typename DataArray::shape_type Shape4Type;

        typedef nifty::array::StaticArray<int64_t, 4> AffCoordType;

        // get the shapes with halos
        std::vector<size_t> beginWithHalo(3), endWithHalo(3), affsBegin(4);
        const auto & volumeShape = labelsDs->shape();
        for(unsigned axis = 0; axis < 3; ++axis) {
            beginWithHalo[axis] = std::max(static_cast<int64_t>(roiBegin[axis]) - static_cast<int64_t>(haloBegin[axis]), 0L);
            endWithHalo[axis] = std::min(roiEnd[axis] + haloEnd[axis], volumeShape[axis]);
            affsBegin[axis + 1] = beginWithHalo[axis];
        }
        // TODO if we want to be more general, we need to allow for non-zero channel offset here
        affsBegin[0] = 0;

        Shape3Type shape;
        Shape4Type affShape;
        CoordType blockShape;
        AffCoordType affBlockShape;
        for(unsigned axis = 0; axis < 3; ++axis) {
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
        z5::multiarray::readSubarray<InputType>(dataDs, affs, affsBegin.begin());
        z5::multiarray::readSubarray<NodeType>(labelsDs, labels, beginWithHalo.begin());

        // create nifty accumulator vector
        AccumulatorVector accumulators(graph.numberOfEdges());
        const int pass = 1;
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        const bool byteInput = typeid(InputType) == typeid(uint8_t);

        // We need different accumulation for byte and float input
        if(byteInput) {
            // accumulate
            nifty::tools::forEachCoordinate(affBlockShape, [&](const AffCoordType & affCoord) {

                CoordType coord, coord2;
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

                const NodeType lU = xtensor::read(labels, coord.asStdArray());
                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lU != lV){
                    // for long range affinites, the uv pair may not be part of the region graph
                    // so we need to check if the edge actually exists
                    const EdgeIndexType edge = graph.findEdge(lU, lV);
                    if(edge != -1) {
                        uint8_t data = xtensor::read(affs, affCoord.asStdArray());
                        FeatureType dataF = static_cast<FeatureType>(data) / 255.;
                        accumulators[edge].updatePassN(dataF, pass);
                    }
                }
            });
        }
        else {
            // accumulate
            nifty::tools::forEachCoordinate(affBlockShape, [&](const AffCoordType & affCoord) {

                CoordType coord, coord2;
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

                const NodeType lU = xtensor::read(labels, coord.asStdArray());
                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lU != lV){
                    // for long range affinites, the uv pair may not be part of the region graph
                    // so we need to check if the edge actually exists
                    const EdgeIndexType edge = graph.findEdge(lU, lV);
                    if(edge != -1) {
                        accumulators[edge].updatePassN(xtensor::read(affs, affCoord.asStdArray()), pass);
                    }
                }
            });

        }

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, blockStoragePath);
    }


    ///
    // bind specific accumulators
    ///


    template<class InputType>
    inline void extractBlockFeaturesFromBoundaryMaps(const std::string & blockPrefix,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<size_t> & blockIds,
                                                     const std::string & tmpFeatureStorage,
                                                     FeatureType dataMin=0, FeatureType dataMax=1) {

        // TODO could also use the std::bind pattern and std::function
        auto accumulator = [dataMin, dataMax](
                const Graph & graph,
                std::unique_ptr<z5::Dataset> dataDs,
                std::unique_ptr<z5::Dataset> labelsDs,
                const std::vector<size_t> & roiBegin,
                const std::vector<size_t> & roiEnd,
                const std::string & blockStoragePath) {

            accumulateBoundaryMap<InputType>(graph, std::move(dataDs), std::move(labelsDs),
                                             roiBegin, roiEnd, blockStoragePath,
                                             dataMin, dataMax);
        };

        extractBlockFeaturesImpl(blockPrefix,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, tmpFeatureStorage,
                                 accumulator);
    }


    template<class InputType>
    inline void extractBlockFeaturesFromAffinityMaps(const std::string & blockPrefix,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<size_t> & blockIds,
                                                     const std::string & tmpFeatureStorage,
                                                     const std::vector<OffsetType> & offsets,
                                                     FeatureType dataMin=0, FeatureType dataMax=1) {
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

            accumulateAffinityMap<InputType>(graph, std::move(dataDs), std::move(labelsDs),
                                             roiBegin, roiEnd, blockStoragePath,
                                             offsets, haloBegin, haloEnd,
                                             dataMin, dataMax);
        };

        extractBlockFeaturesImpl(blockPrefix,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, tmpFeatureStorage,
                                 accumulator);
    }


    ///
    // Feature block merging
    ///


    inline void loadBlockFeatures(const std::string & blockFeaturePath,
                                  xt::xtensor<FeatureType, 2> & features) {
        const std::vector<size_t> zero2Coord({0, 0});
        auto featDs = z5::openDataset(blockFeaturePath);
        z5::multiarray::readSubarray<FeatureType>(featDs, features, zero2Coord.begin());
    }


    inline void mergeFeaturesForSingleEdge(const xt::xtensor<FeatureType, 2> & tmpFeatures, xt::xtensor<FeatureType, 2> & targetFeatures,
                                           const EdgeIndexType tmpId, const EdgeIndexType targetId,
                                           std::vector<bool> & hasFeatures) {

        if(hasFeatures[targetId]) {

            // index 9 is the count
            const FeatureType nSamplesA = targetFeatures(targetId, 9);
            const FeatureType nSamplesB = tmpFeatures(tmpId, 9);
            const FeatureType nSamplesTot = nSamplesA + nSamplesB;
            const FeatureType ratioA = nSamplesA / nSamplesTot;
            const FeatureType ratioB = nSamplesB / nSamplesTot;

            // merge the mean
            const FeatureType meanA = targetFeatures(targetId, 0);
            const FeatureType meanB = tmpFeatures(tmpId, 0);
            const FeatureType newMean = ratioA * meanA + ratioB * meanB;
            targetFeatures(targetId, 0) = newMean;

            // merge the variance (feature id 1)
            // see https://stackoverflow.com/questions/1480626/merging-two-statistical-result-sets
            const FeatureType varA = targetFeatures(targetId, 1);
            const FeatureType varB = tmpFeatures(tmpId, 1);
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

        //// debugging
        //for(auto featId = 0; featId < 10; ++featId) {
        //    if(std::isinf(targetFeatures(targetId, featId)) || std::isnan(targetFeatures(targetId, featId))) {
        //        std::cout << "Feat is nan /  inf " << targetFeatures(targetId, featId) << std::endl;
        //        std::cout << "For feat " << targetId << " " << featId << std::endl;
        //        throw std::runtime_error("NaNNaNNaNNaNNaNNaNNaNNaNNaNNaN Batman");
        //    }
        //}
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
            xt::xtensor<FeatureType, 2> features;
            xt::xtensor<FeatureType, 2> tmpFeatures;
            std::vector<bool> edgeHasFeatures;
        };
        std::vector<PerThreadData> perThreadDataVector(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&](const int t, const int tId){
            auto & ptd = perThreadDataVector[tId];
            ptd.features = xt::xtensor<FeatureType, 2>(fShape);
            ptd.tmpFeatures = xt::xtensor<FeatureType, 2>(tmpInit);
            ptd.edgeHasFeatures = std::vector<bool>(nEdges, false);
        });

        // iterate over the block ids
        const size_t nBlocks = blockIds.size();
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int tId, const int blockIndex){

            const size_t blockId = blockIds[blockIndex];

            auto & perThreadData = perThreadDataVector[tId];
            auto & features = perThreadData.features;
            auto & hasFeatures = perThreadData.edgeHasFeatures;
            auto & blockFeatures = perThreadData.tmpFeatures;

            // load edge ids for the block
            const std::string blockGraphPath = graphBlockPrefix + std::to_string(blockId);
            std::vector<EdgeIndexType> blockEdgeIndices;
            loadEdgeIndices(blockGraphPath, blockEdgeIndices, 0);
            const size_t nEdgesBlock = blockEdgeIndices.size();

            // get mapping to dense edge ids for this block
            std::unordered_map<EdgeIndexType, EdgeIndexType> toDenseBlockId;
            size_t denseBlockId = 0;
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                toDenseBlockId[edgeId] = denseBlockId;
                ++denseBlockId;
            }

            // load features for the block
            // first resize the tmp features, if necessary
            const std::string blockFeaturePath = featureBlockPrefix + std::to_string(blockId);
            if(blockFeatures.shape()[0] != nEdgesBlock) {
                blockFeatures.resize({nEdgesBlock, 10});
            }

            loadBlockFeatures(blockFeaturePath, blockFeatures);

            // iterate over the edges in this block and merge edge features if they are in our edge range
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                // only merge if we are in the edge range
                if(edgeId >= edgeIdBegin && edgeId < edgeIdEnd) {
                    // find the corresponding ids in the dense block edges
                    // and in the dense edge range
                    const EdgeIndexType rangeEdgeId = edgeId - edgeIdBegin;
                    const EdgeIndexType blockEdgeId = toDenseBlockId[edgeId];

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
                    mergeFeaturesForSingleEdge(perThreadData.features, features, edgeId, edgeId, hasFeatureVector);
                }
            }
        });

        // TODO we could parallelize this over the out chunks
        // serialize the edge features
        auto dsOut = z5::openDataset(featuresOut);
        const std::vector<size_t> featOffset({edgeIdBegin, 0});
        z5::multiarray::writeSubarray<FeatureType>(dsOut, features, featOffset.begin());
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
            bool haveEdges = loadEdgeIndices(blockPath, blockEdgeIndices, 0);
            if(!haveEdges) {
                return;
            }

            // first we check, if the block has overlap with at least one of our edges
            // (and thus if it is relevant) by a simple range check
            // this is a bit tricky, because we have a dense input edge id range,
            // however the id range in the block is not dense.
            // hence, we project to a dense range in the block, which is fine for 
            // just determining whether we have overlap

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

        // find all the blocks that contain edges in the current range
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
