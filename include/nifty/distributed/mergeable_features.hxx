#pragma once

#include "vigra/accumulator.hxx"
#include "nifty/distributed/distributed_graph.hxx"

#ifdef WITH_BOOST_FS
    namespace fs = boost::filesystem;
#else
    #if __GCC__ > 7
        namespace fs = std::filesystem;
    #else
        namespace fs = std::experimental::filesystem;
    #endif
#endif
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


    typedef nifty::array::StaticArray<int64_t, 4> AffCoordType;
    typedef std::array<int, 3> OffsetType;


    template<class T,class U>
    inline T replaceIfNotFinite(const T & val, const U & replaceVal){
        if(std::isfinite(val))
            return val;
        else
            return replaceVal;
    }


    template<class FEATURE_ACCUMULATOR>
    inline void extractBlockFeaturesImpl(const std::string & graphPath,
                                         const std::string & subgraphKey,
                                         const std::string & dataPath,
                                         const std::string & dataKey,
                                         const std::string & labelPath,
                                         const std::string & labelKey,
                                         const std::vector<std::size_t> & blockIds,
                                         FEATURE_ACCUMULATOR && accumulator) {

        z5::filesystem::handle::File dataFile(dataPath);
        z5::filesystem::handle::File labelFile(labelPath);
        z5::filesystem::handle::File graphFile(graphPath);

        // get ignore label flag from the attributes of the sub-graph group
        z5::filesystem::handle::Group subgraphGroup(graphFile, subgraphKey);
        nlohmann::json j;
        z5::readAttributes(subgraphGroup, j);
        const bool ignoreLabel = j["ignore_label"];

        // open the edge dataset
        const auto dsEdges = z5::openDataset(subgraphGroup, "edges");

        const auto & blocking = dsEdges->chunking();
        std::vector<std::size_t> blockPos(3), roiBegin(3), roiEnd(3);
        for(const auto blockId : blockIds) {

            // we move the unique ptr, so after the loop it will be null
            // hence we need to create the unique-ptr in each loop again.
            // (the proper way to do this would be with shared_ptrs,
            //  but for this some of the API needs to change)
            auto data = z5::openDataset(dataFile, dataKey);
            auto labels = z5::openDataset(labelFile, labelKey);

            blocking.blockIdToBlockCoordinate(blockId, blockPos);
            // continue if we don't have edges in this block
            if(!dsEdges->chunkExists(blockPos)) {
                continue;
            }

            // load the edges and build the graph
            std::size_t nEdges;
            dsEdges->checkVarlenChunk(blockPos, nEdges);
            std::vector<uint64_t> edgeSer(nEdges);
            dsEdges->readChunk(blockPos, &edgeSer[0]);
            xt::xtensor<uint64_t, 2> edges({nEdges / 2, 2});
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                edges(edgeId, 0) = edgeSer[2 * edgeId];
                edges(edgeId, 1) = edgeSer[2 * edgeId + 1];
            }
            Graph graph(edges);

            // load the bounding box
            blocking.getBlockBeginAndEnd(blockPos, roiBegin, roiEnd);

            // run the accumulator
            accumulator(graph, std::move(data), std::move(labels),
                        blockPos, roiBegin, roiEnd, blockId, ignoreLabel);
        }
    }


    ///
    // accumulator implementations
    ///


    // helper function to serialize edge features
    // unfortunately, we can't (trivially) serialize the state of the accumulators
    // so instead, we get the desired statistics and serialize those, to be merged
    // later heuristically
    inline void serializeDefaultEdgeFeatures(const AccumulatorVector & accumulators,
                                             const std::string & outPath,
                                             const std::string & outKey,
                                             const std::vector<std::size_t> & chunkPos) {
        // the number of features is hard-coded to 10 for now
        // we write out the flat value vector, so we don't bother arranging features in 2d
        const std::size_t n_features = 10 * accumulators.size();
        std::vector<FeatureType> values(n_features);

        std::size_t featId = 0;
        for(const auto & accumulator : accumulators) {

            // mean and variance
            const FeatureType mean = replaceIfNotFinite(acc::get<acc::Mean>(accumulator), 0.0);
            values[featId] = mean;
            ++featId;
            values[featId] = replaceIfNotFinite(acc::get<acc::Variance>(accumulator), 0.0);
            ++featId;

            // quantiles
            const auto & quantiles = acc::get<Quantiles>(accumulator);
            for(unsigned qi = 0; qi < 7; ++qi) {
                values[featId] = replaceIfNotFinite(quantiles[qi], mean);
                ++featId;
            }

            // count
            values[featId] = replaceIfNotFinite(acc::get<acc::Count>(accumulator), 0.0);
            ++featId;
        }

        // open the output dataset and write the features
        const z5::filesystem::handle::File outFile(outPath);
        auto ds = z5::openDataset(outFile, outKey);
        ds->writeChunk(chunkPos, &values[0], true, values.size());
    }


    template<class INPUT, class LABELS>
    inline void accumulateBoundariesImplByte(const Graph & graph,
                                             const xt::xtensor<INPUT, 3> & data,
                                             const xt::xtensor<LABELS, 3> & labels,
                                             const CoordType & blockShape,
                                             const bool ignoreLabel,
                                             const std::array<bool, 3> & increaseRoi,
                                             AccumulatorVector & accumulators) {
        const int pass = 1;
        const bool anyIncreased = std::any_of(increaseRoi.begin(), increaseRoi.end(), [](const bool val){return val;});
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {
            const NodeType lU = xtensor::read(labels, coord.asStdArray());
            if(lU == 0 && ignoreLabel) {
                return;
            }

            // check if we need to skip any axes due to increase roi
            std::array<bool, 3> skipAxis = {false, false, false};
            if(anyIncreased) {

                // check if we have any axis on the face
                std::array<bool, 3> onFace = {false, false, false};
                for(std::size_t axis = 0; axis < 3; ++axis){
                    onFace[axis] = (coord[axis] == 0) && increaseRoi[axis];
                }

                // check how many axes are on the face
                const int faceSum = std::accumulate(onFace.begin(), onFace.end(), 0, std::plus<int>());

                // depending on the number of axes on the face:
                //  0: we don't need to skip any axis
                //  1: we are on a face -> we need to skip the axes not on the face so we don't overcount
                // >1: we are on a line or point ->  we need to skip all axes, because all adjacent nodes are on a face
                if(faceSum == 1) {
                    for(std::size_t axis = 0; axis < 3; ++axis){
                        skipAxis[axis] = !onFace[axis];
                    }
                } else if(faceSum > 1) {
                    return;
                }
            }

            CoordType coord2;
            for(std::size_t axis = 0; axis < 3; ++axis){

                if(skipAxis[axis]){
                    continue;
                }

                makeCoord2(coord, coord2, axis);
                if(coord2[axis] >= blockShape[axis]){
                    continue;
                }

                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lV == 0 && ignoreLabel) {
                    continue;
                }
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
        });
    }


    template<class INPUT, class LABELS>
    inline void accumulateBoundariesImplFloat(const Graph & graph,
                                              const INPUT & data,
                                              const LABELS & labels,
                                              const CoordType & blockShape,
                                              const bool ignoreLabel,
                                              const std::array<bool, 3> & increaseRoi,
                                              AccumulatorVector & accumulators) {
        const int pass = 1;
        const bool anyIncreased = std::any_of(increaseRoi.begin(), increaseRoi.end(), [](const bool val){return val;});
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {
            const NodeType lU = xtensor::read(labels, coord.asStdArray());
            if(lU == 0 && ignoreLabel) {
                return;
            }

            // check if we need to skip any axes due to increase roi
            std::array<bool, 3> skipAxis = {false, false, false};
            if(anyIncreased) {

                // check if we have any axis on the face
                std::array<bool, 3> onFace = {false, false, false};
                for(std::size_t axis = 0; axis < 3; ++axis){
                    onFace[axis] = (coord[axis] == 0) && increaseRoi[axis];
                }

                // check how many axes are on the face
                const int faceSum = std::accumulate(onFace.begin(), onFace.end(), 0, std::plus<int>());

                // depending on the number of axes on the face:
                //  0: we don't need to skip any axis
                //  1: we are on a face -> we need to skip the axes not on the face so we don't overcount
                // >1: we are on a line or point ->  we need to skip all axes, because all adjacent nodes are on a face
                if(faceSum == 1) {
                    for(std::size_t axis = 0; axis < 3; ++axis){
                        skipAxis[axis] = !onFace[axis];
                    }
                } else if(faceSum > 1) {
                    return;
                }
            }

            CoordType coord2;
            for(std::size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);
                if(coord2[axis] >= blockShape[axis]){
                    continue;
                }
                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lV == 0 && ignoreLabel) {
                    continue;
                }
                if(lU != lV){
                    const EdgeIndexType edge = graph.findEdge(lU, lV);
                    const auto fU = xtensor::read(data, coord.asStdArray());
                    const auto fV = xtensor::read(data, coord2.asStdArray());
                    FeatureType fUf = static_cast<FeatureType>(fU);
                    FeatureType fVf = static_cast<FeatureType>(fV);
                    accumulators[edge].updatePassN(fUf, pass);
                    accumulators[edge].updatePassN(fVf, pass);
                }
            }
        });
    }


    // accumulate simple boundary map
    template<class InputType>
    inline void accumulateBoundaryMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<std::size_t> & chunkPos,
                                      const std::vector<std::size_t> & roiBegin,
                                      const std::vector<std::size_t> & roiEnd,
                                      const std::string & outPath,
                                      const std::string & outKey,
                                      const FeatureType dataMin,
                                      const FeatureType dataMax,
                                      const bool ignoreLabel,
                                      const bool increaseRoi=false,
                                      const bool isLabelMultiset=false) {
        // xtensor typedefs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<InputType, 3> DataArray;

        // check if we need to increase the roi
        // if specified, we decrease roiBegin by 1.
        // to match what was done in the graph extraction when increaseRoi is true
        std::vector<std::size_t> actualRoiBegin = roiBegin;
        std::array<bool, 3> increaseRoiArray = {false, false, false};
        if(increaseRoi) {
            for(int axis = 0; axis < 3; ++axis) {
                if(actualRoiBegin[axis] > 0) {
                    --actualRoiBegin[axis];
                    increaseRoiArray[axis] = true;
                }
            }
        }

        // get the shapes
        Shape3Type shape;
        CoordType blockShape;
        for(unsigned axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - actualRoiBegin[axis];
            blockShape[axis] = shape[axis];
        }

        // load data and labels
        DataArray data(shape);
        z5::multiarray::readSubarray<InputType>(dataDs, data, actualRoiBegin.begin());

        // load labels from normal label dataset or label multi-set
        LabelArray labels(shape);
        if(isLabelMultiset) {
            // is a label multiset -> need to use label multi-set wrapper and then load the array
            tools::LabelMultisetWrapper label_multiset(std::move(labelsDs));
            label_multiset.readSubarray(labels, actualRoiBegin);
        } else {
            z5::multiarray::readSubarray<NodeType>(labelsDs, labels, actualRoiBegin.begin());
        }

        // create nifty accumulator vector
        AccumulatorVector accumulators(graph.numberOfEdges());
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        const bool byteInput = typeid(InputType) == typeid(uint8_t);

        // accumulate
        if(byteInput) {
            accumulateBoundariesImplByte(graph, data, labels, blockShape,
                                         ignoreLabel, increaseRoiArray, accumulators);
        } else {
            accumulateBoundariesImplFloat(graph, data, labels, blockShape,
                                          ignoreLabel, increaseRoiArray, accumulators);
        }

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, outPath, outKey, chunkPos);
    }


    template<class AFFS, class LABELS>
    inline void accumulateAffinitiesImplByte(const Graph & graph,
                                             const AFFS & affs,
                                             const LABELS & labels,
                                             const AffCoordType & affBlockShape,
                                             const std::vector<OffsetType> & offsets,
                                             const bool ignoreLabel,
                                             AccumulatorVector & accumulators) {
        const int pass = 1;
        // accumulate
        nifty::tools::forEachCoordinate(affBlockShape, [&](const AffCoordType & affCoord) {

            CoordType coord, coord2;
            // the 0th affinitiy coordinate gives the channel index
            const auto & offset = offsets[affCoord[0]];
            for(unsigned axis = 0; axis < 3; ++axis) {
                coord[axis] = affCoord[axis + 1];
                coord2[axis] = affCoord[axis + 1] + offset[axis];

                // bounds check
                if(coord2[axis] < 0 || coord2[axis] > affBlockShape[axis + 1]) {
                    return;
                }
            }

            const NodeType lU = xtensor::read(labels, coord.asStdArray());
            const NodeType lV = xtensor::read(labels, coord2.asStdArray());

            if(ignoreLabel && (lU == 0 || lV == 0)) {
                return;
            }

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


    template<class AFFS, class LABELS>
    inline void accumulateAffinitiesImplFloat(const Graph & graph,
                                              const AFFS & affs,
                                              const LABELS & labels,
                                              const AffCoordType & affBlockShape,
                                              const std::vector<OffsetType> & offsets,
                                              const bool ignoreLabel,
                                              AccumulatorVector & accumulators) {
        // accumulate
        const int pass = 1;
        nifty::tools::forEachCoordinate(affBlockShape, [&](const AffCoordType & affCoord) {

            CoordType coord, coord2;
            // the 0th affinitiy coordinate gives the channel index
            const auto & offset = offsets[affCoord[0]];
            for(unsigned axis = 0; axis < 3; ++axis) {
                coord[axis] = affCoord[axis + 1];
                coord2[axis] = affCoord[axis + 1] + offset[axis];

                // bounds check
                if(coord2[axis] < 0 || coord2[axis] > affBlockShape[axis + 1]) {
                    return;
                }
            }

            const NodeType lU = xtensor::read(labels, coord.asStdArray());
            const NodeType lV = xtensor::read(labels, coord2.asStdArray());

            if(ignoreLabel && (lU == 0 || lV == 0)) {
                return;
            }

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


    // accumulate affinity maps
    template<class InputType>
    inline void accumulateAffinityMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<std::size_t> & chunkPos,
                                      const std::vector<std::size_t> & roiBegin,
                                      const std::vector<std::size_t> & roiEnd,
                                      const std::string & outPath,
                                      const std::string & outKey,
                                      const std::vector<OffsetType> & offsets,
                                      const std::vector<std::size_t> & haloBegin,
                                      const std::vector<std::size_t> & haloEnd,
                                      const FeatureType dataMin,
                                      const FeatureType dataMax,
                                      const bool ignoreLabel,
                                      const bool isLabelMultiset) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<InputType, 4> DataArray;
        typedef typename DataArray::shape_type Shape4Type;

        // get the shapes with halos
        std::vector<std::size_t> beginWithHalo(3), endWithHalo(3), affsBegin(4);
        const auto & volumeShape = labelsDs->shape();
        for(unsigned axis = 0; axis < 3; ++axis) {
            beginWithHalo[axis] = std::max(static_cast<int64_t>(roiBegin[axis]) - static_cast<int64_t>(haloBegin[axis]), static_cast<int64_t>(0));
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

        // load data
        DataArray affs(affShape);
        z5::multiarray::readSubarray<InputType>(dataDs, affs, affsBegin.begin());

        // load labels from normal label dataset or label multi-set
        LabelArray labels(shape);
        if(isLabelMultiset) {
            // is a label multiset -> need to use label multi-set wrapper and then load the array
            tools::LabelMultisetWrapper label_multiset(std::move(labelsDs));
            label_multiset.readSubarray(labels, beginWithHalo);
        } else {
            z5::multiarray::readSubarray<NodeType>(labelsDs, labels, beginWithHalo.begin());
        }

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
            accumulateAffinitiesImplByte(graph, affs, labels,
                                         affBlockShape, offsets,
                                         ignoreLabel, accumulators);
        }
        else {
            accumulateAffinitiesImplFloat(graph, affs, labels,
                                          affBlockShape, offsets,
                                          ignoreLabel, accumulators);
        }

        // serialize the accumulators
        serializeDefaultEdgeFeatures(accumulators, outPath, outKey, chunkPos);
    }


    ///
    // bind specific accumulators
    ///


    template<class InputType>
    inline void extractBlockFeaturesFromBoundaryMaps(const std::string & graphPath,
                                                     const std::string & subgraphKey,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<std::size_t> & blockIds,
                                                     const std::string & outPath,
                                                     const std::string & outKey,
                                                     const FeatureType dataMin=0,
                                                     const FeatureType dataMax=1,
                                                     const bool increaseRoi=false) {

        // check if we have label multiset input
        z5::filesystem::handle::File fileHandle(labelPath);
        z5::filesystem::handle::Dataset dsHandle(fileHandle, labelKey);
        nlohmann::json attrs;
        z5::readAttributes(dsHandle, attrs);

        bool isLabelMultiset = false;
        if(attrs.find("isLabelMultiset") != attrs.end()) {
            isLabelMultiset = attrs["isLabelMultiset"];
        }

        auto accumulator = [dataMin, dataMax, increaseRoi, isLabelMultiset,
                            &outPath, &outKey](
                const Graph & graph,
                std::unique_ptr<z5::Dataset> dataDs,
                std::unique_ptr<z5::Dataset> labelsDs,
                const std::vector<std::size_t> & chunkPos,
                const std::vector<std::size_t> & roiBegin,
                const std::vector<std::size_t> & roiEnd,
                const std::size_t blockId,
                const bool ignoreLabel) {

            accumulateBoundaryMap<InputType>(graph, std::move(dataDs), std::move(labelsDs),
                                             chunkPos, roiBegin, roiEnd, outPath, outKey,
                                             dataMin, dataMax, ignoreLabel,
                                             increaseRoi, isLabelMultiset);
        };

        extractBlockFeaturesImpl(graphPath, subgraphKey,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, accumulator);
    }


    template<class InputType>
    inline void extractBlockFeaturesFromAffinityMaps(const std::string & graphPath,
                                                     const std::string & subgraphKey,
                                                     const std::string & dataPath,
                                                     const std::string & dataKey,
                                                     const std::string & labelPath,
                                                     const std::string & labelKey,
                                                     const std::vector<std::size_t> & blockIds,
                                                     const std::string & outPath,
                                                     const std::string & outKey,
                                                     const std::vector<OffsetType> & offsets,
                                                     const FeatureType dataMin=0,
                                                     const FeatureType dataMax=1) {
        // calculate max halos from the offsets
        std::vector<std::size_t> haloBegin(3), haloEnd(3);
        for(const auto & offset : offsets) {
            for(unsigned axis = 0; axis < 3; ++axis) {
                if(offset[axis] < 0) {
                    haloBegin[axis] = std::max(static_cast<std::size_t>(-offset[axis]), haloBegin[axis]);
                } else if(offset[axis] > 0) {
                    haloEnd[axis] = std::max(static_cast<std::size_t>(offset[axis]), haloEnd[axis]);
                }
            }
        }

        // check if we have label multiset input
        z5::filesystem::handle::File fileHandle(labelPath);
        z5::filesystem::handle::Dataset dsHandle(fileHandle, labelKey);
        nlohmann::json attrs;
        z5::readAttributes(dsHandle, attrs);

        bool isLabelMultiset = false;
        if(attrs.find("isLabelMultiset") != attrs.end()) {
            isLabelMultiset = attrs["isLabelMultiset"];
        }

        auto accumulator = [dataMin, dataMax, isLabelMultiset,
                            &offsets, &haloBegin, &haloEnd,
                            &outPath, &outKey](
                const Graph & graph,
                std::unique_ptr<z5::Dataset> dataDs,
                std::unique_ptr<z5::Dataset> labelsDs,
                const std::vector<std::size_t> & chunkPos,
                const std::vector<std::size_t> & roiBegin,
                const std::vector<std::size_t> & roiEnd,
                const std::size_t blockId,
                const bool ignoreLabel) {

            accumulateAffinityMap<InputType>(graph, std::move(dataDs), std::move(labelsDs),
                                             chunkPos, roiBegin, roiEnd, outPath, outKey,
                                             offsets, haloBegin, haloEnd,
                                             dataMin, dataMax, ignoreLabel,
                                             isLabelMultiset);
        };

        extractBlockFeaturesImpl(graphPath, subgraphKey,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, accumulator);
    }


    ///
    // Feature block merging
    ///


    inline void mergeFeatType(const xt::xtensor<FeatureType, 2> & tmpFeatures,
                              xt::xtensor<FeatureType, 2> & targetFeatures,
                              const EdgeIndexType tmpId,  const EdgeIndexType targetId,
                              const unsigned off,
                              const FeatureType ratioA, const FeatureType ratioB) {
        // merge the mean
        const FeatureType meanA = targetFeatures(targetId, off);
        const FeatureType meanB = tmpFeatures(tmpId, off);
        const FeatureType mean = replaceIfNotFinite(ratioA * meanA + ratioB * meanB, 0);
        targetFeatures(targetId, off) = mean;

        // merge the variance (feature id 1), see
        // https://stackoverflow.com/questions/1480626/merging-two-statistical-result-sets
        const FeatureType varA = targetFeatures(targetId, off + 1);
        const FeatureType varB = tmpFeatures(tmpId, off + 1);
        const FeatureType var = ratioA * (varA + (meanA - mean) * (meanA - mean)) +
                                ratioB * (varB + (meanB - mean) * (meanB - mean));
        targetFeatures(targetId, off + 1) = replaceIfNotFinite(var, 0);

        // merge the min (feature id 2)
        const FeatureType min = std::min(targetFeatures(targetId, off + 2),
                                         tmpFeatures(tmpId, off + 2));
        targetFeatures(targetId, off + 2) = replaceIfNotFinite(min, mean);

        // merge the quantiles (not min and max !) via weighted average
        // this is not correct, but the best we can do for now
        for(std::size_t featId = 3 + off; featId < 8 + off; ++featId) {
            const FeatureType quant = ratioA * targetFeatures(targetId, featId) +
                                      ratioB * tmpFeatures(tmpId, featId);
            targetFeatures(targetId, featId) = replaceIfNotFinite(quant, mean);
        }

        // merge the max (feature id 8)
        const FeatureType max = std::max(targetFeatures(targetId, off + 8),
                                                        tmpFeatures(tmpId, off + 8));
        targetFeatures(targetId, off + 8) = replaceIfNotFinite(max, mean);
    }


    inline void mergeFeaturesForSingleEdge(const xt::xtensor<FeatureType, 2> & tmpFeatures,
                                           xt::xtensor<FeatureType, 2> & targetFeatures,
                                           const EdgeIndexType tmpId,
                                           const EdgeIndexType targetId,
                                           std::vector<bool> & hasFeatures) {

        const std::size_t nFeatures = tmpFeatures.shape()[1];
        const std::size_t featsPerType = 9;
        if(hasFeatures[targetId]) {

            // last index is the count
            const FeatureType nSamplesA = targetFeatures(targetId, nFeatures - 1);
            const FeatureType nSamplesB = tmpFeatures(tmpId, nFeatures - 1);
            const FeatureType nSamplesTot = nSamplesA + nSamplesB;
            const FeatureType ratioA = nSamplesA / nSamplesTot;
            const FeatureType ratioB = nSamplesB / nSamplesTot;

            // merge each type of features in 9er steps
            const std::size_t nFeatTypes = (nFeatures - 1) / featsPerType;
            for(std::size_t featType = 0; featType < nFeatTypes; ++featType) {
                mergeFeatType(tmpFeatures, targetFeatures,
                              tmpId, targetId, featType * featsPerType,
                              ratioA, ratioB);
            }

            // merge the count (last feature index)
            targetFeatures(targetId, nFeatures - 1) = nSamplesTot;

        } else {

            for(std::size_t featId = 0; featId < nFeatures; ++featId) {
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


    inline void mergeEdgeFeaturesForBlocks(const std::string & graphPath,
                                           const std::string & graphKey,
                                           const std::string & inPath,
                                           const std::string & inKey,
                                           const std::size_t edgeIdBegin,
                                           const std::size_t edgeIdEnd,
                                           const std::size_t nFeatures,
                                           const std::vector<std::size_t> & blockIds,
                                           nifty::parallel::ThreadPool & threadpool,
                                           const std::string & outPath,
                                           const std::string & outKey) {
        //
        const std::size_t nEdges = edgeIdEnd - edgeIdBegin;
        Shape2Type fShape = {nEdges, nFeatures};
        Shape2Type tmpInit = {1, nFeatures};

        // initialize per thread data
        const std::size_t nThreads = threadpool.nThreads();
        struct PerThreadData {
            xt::xtensor<FeatureType, 2> features;
            std::vector<bool> edgeHasFeatures;
        };
        std::vector<PerThreadData> perThreadDataVector(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&](const int t, const int tId){
            auto & ptd = perThreadDataVector[tId];
            ptd.features = xt::xtensor<FeatureType, 2>(fShape);
            ptd.edgeHasFeatures = std::vector<bool>(nEdges, false);
        });

        const z5::filesystem::handle::File graphFile(graphPath);
        const z5::filesystem::handle::File inFile(inPath);

        // load the edge id dataset
        const std::string edgeIdKey = graphKey + "/edge_ids";
        const auto dsEdgeIds = z5::openDataset(graphFile, edgeIdKey);
        const auto & blocking = dsEdgeIds->chunking();

        // load the input feature dataset
        const auto dsIn = z5::openDataset(inFile, inKey);

        // iterate over the block ids
        const std::size_t nBlocks = blockIds.size();
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int tId,
                                                                   const int blockIndex){

            const std::size_t blockId = blockIds[blockIndex];
            std::vector<std::size_t> chunkPos(3);
            blocking.blockIdToBlockCoordinate(blockId, chunkPos);

            auto & perThreadData = perThreadDataVector[tId];
            auto & features = perThreadData.features;
            auto & hasFeatures = perThreadData.edgeHasFeatures;

            // load edge ids for the block
            std::size_t nEdgesBlock;
            dsEdgeIds->checkVarlenChunk(chunkPos, nEdgesBlock);
            std::vector<EdgeIndexType> blockEdgeIndices(nEdgesBlock);
            dsEdgeIds->readChunk(chunkPos, &blockEdgeIndices[0]);

            // get mapping to dense edge ids for this block
            std::unordered_map<EdgeIndexType, EdgeIndexType> toDenseBlockId;
            std::size_t denseBlockId = 0;
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                toDenseBlockId[edgeId] = denseBlockId;
                ++denseBlockId;
            }

            // load features for the block
            const std::size_t nFeatsBlock = nEdgesBlock * nFeatures;
            std::vector<FeatureType> tmpf(nFeatsBlock);
            dsIn->readChunk(chunkPos, &tmpf[0]);

            // adapt to xtensor
            auto blockFeatures = xt::adapt(tmpf, {nEdgesBlock, nFeatures});

            // iterate over the edges in this block and merge edge features
            // if they are in our edge range
            for(EdgeIndexType edgeId : blockEdgeIndices) {
                // only merge if we are in the edge range
                if(edgeId >= edgeIdBegin && edgeId < edgeIdEnd) {
                    // find the corresponding ids in the dense block edges
                    // and in the dense edge range
                    const EdgeIndexType rangeEdgeId = edgeId - edgeIdBegin;
                    const EdgeIndexType blockEdgeId = toDenseBlockId[edgeId];

                    mergeFeaturesForSingleEdge(blockFeatures, features,
                                               blockEdgeId, rangeEdgeId, hasFeatures);
                }
            }

        });

        // merge features for each edge
        auto & features = perThreadDataVector[0].features;
        auto & hasFeatureVector = perThreadDataVector[0].edgeHasFeatures;
        nifty::parallel::parallel_foreach(threadpool, nEdges, [&](const int tId,
                                                                  const EdgeIndexType edgeId){
            for(int threadId = 1; threadId < nThreads; ++threadId) {
                auto & perThreadData = perThreadDataVector[threadId];
                if(perThreadData.edgeHasFeatures[edgeId]) {
                    mergeFeaturesForSingleEdge(perThreadData.features, features,
                                               edgeId, edgeId, hasFeatureVector);
                }
            }
        });

        // serialize the edge features
        const z5::filesystem::handle::File outFile(outPath);
        auto dsOut = z5::openDataset(outFile, outKey);
        const std::vector<std::size_t> featOffset({edgeIdBegin, 0});
        z5::multiarray::writeSubarray<FeatureType>(dsOut, features, featOffset.begin(), nThreads);
    }


    inline void findRelevantBlocks(const std::string & graphPath, const std::string & graphKey,
                                   const std::vector<std::size_t> & blockIds,
                                   const std::size_t edgeIdBegin,
                                   const std::size_t edgeIdEnd,
                                   nifty::parallel::ThreadPool & threadpool,
                                   std::vector<std::size_t> & relevantBlocks) {

        const std::size_t nThreads = threadpool.nThreads();
        std::vector<std::set<std::size_t>> perThreadData(nThreads);

        const std::size_t numberOfBlocks = blockIds.size();
        const z5::filesystem::handle::File graphFile(graphPath);
        const std::string edgeIdKey = graphKey + "/edge_ids";
        const auto dsEdgeIds = z5::openDataset(graphFile, edgeIdKey);
        const auto & blocking = dsEdgeIds->chunking();

        nifty::parallel::parallel_foreach(threadpool, numberOfBlocks, [&](const int tId,
                                                                          const int blockIndex) {
            const std::size_t blockId = blockIds[blockIndex];
            std::vector<std::size_t> chunkPos(3);
            blocking.blockIdToBlockCoordinate(blockId, chunkPos);

            // read the edge ids from this chunk
            if(!dsEdgeIds->chunkExists(chunkPos)) {
                return;
            }

            std::size_t nEdges;
            dsEdgeIds->checkVarlenChunk(chunkPos, nEdges);
            std::vector<EdgeIndexType> blockEdgeIndices(nEdges);
            dsEdgeIds->readChunk(chunkPos, &blockEdgeIndices[0]);

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
        auto & relevantBlocksSet = perThreadData[0];
        for(int tId = 1; tId < nThreads; ++tId) {
            relevantBlocksSet.insert(perThreadData[tId].begin(), perThreadData[tId].end());
        }

        // write to the out vector
        relevantBlocks.resize(relevantBlocksSet.size());
        std::copy(relevantBlocksSet.begin(), relevantBlocksSet.end(), relevantBlocks.begin());
    }


    inline void mergeFeatureBlocks(const std::string & graphPath,
                                   const std::string & graphKey,
                                   const std::string & inPath,
                                   const std::string & inKey,
                                   const std::string & outPath,
                                   const std::string & outKey,
                                   const std::vector<std::size_t> & blockIds,
                                   const std::size_t edgeIdBegin,
                                   const std::size_t edgeIdEnd,
                                   const int numberOfThreads=1) {
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);

        // find all the blocks that contain edges in the current range
        std::vector<std::size_t> relevantBlocks;
        findRelevantBlocks(graphPath, graphKey, blockIds,
                           edgeIdBegin, edgeIdEnd, threadpool,
                           relevantBlocks);

        // get the number of features
        const z5::filesystem::handle::File outFile(outPath);
        const std::size_t nFeatures = z5::openDataset(outFile, outKey)->shape(1);

        // merge all edges in the edge range
        mergeEdgeFeaturesForBlocks(graphPath, graphKey,
                                   inPath, inKey,
                                   edgeIdBegin, edgeIdEnd, nFeatures,
                                   relevantBlocks, threadpool,
                                   outPath, outKey);
    }


    template<class INPUT, class LABELS, class FEATURES>
    inline void accumulateInput(const Graph & graph,
                                const xt::xexpression<INPUT> & inputExp,
                                const xt::xexpression<LABELS> & labelsExp,
                                const bool ignoreLabel,
                                const bool withSize,
                                const FeatureType dataMin,
                                const FeatureType dataMax,
                                xt::xexpression<FEATURES> & featuresExp) {

        const auto & input = inputExp.derived_cast();
        const auto & labels = labelsExp.derived_cast();
        auto & features = featuresExp.derived_cast();

        AccumulatorVector accumulators(graph.numberOfEdges());
        const int pass = 1;
        HistogramOptions histogramOpts;
        histogramOpts = histogramOpts.setMinMax(dataMin, dataMax);
        for(auto & accumulator : accumulators) {
            accumulator.setHistogramOptions(histogramOpts);
        }

        CoordType shape;
        std::copy(input.shape().begin(), input.shape().end(), shape.begin());

        // TODO we need to get this from the outside
        const std::array<bool, 3> increaseRoiArray = {false, false, false};
        accumulateBoundariesImplFloat(graph, input, labels, shape,
                                      ignoreLabel, increaseRoiArray, accumulators);

        EdgeIndexType edgeId = 0;
        for(const auto & accumulator : accumulators) {
            // get the values from this accumulator
            const FeatureType mean = replaceIfNotFinite(acc::get<acc::Mean>(accumulator), 0.0);

            features(edgeId, 0) = mean;
            features(edgeId, 1) = replaceIfNotFinite(acc::get<acc::Variance>(accumulator), 0.0);
            const auto & quantiles = acc::get<Quantiles>(accumulator);
            for(unsigned qi = 0; qi < 7; ++qi) {
                features(edgeId, 2 + qi) = replaceIfNotFinite(quantiles[qi], mean);
            }
            if(withSize) {
                features(edgeId, 9) = replaceIfNotFinite(acc::get<acc::Count>(accumulator), 0.0);
            }
            ++edgeId;
        }
    }

}
}
