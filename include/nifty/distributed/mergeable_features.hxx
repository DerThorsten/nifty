#pragma once

#include "vigra/accumulator.hxx"
#include "nifty/distributed/distributed_graph.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {


    ///
    // accumulator typedges
    ///
    namespace acc = vigra::acc;
    typedef acc::UserRangeHistogram<32> FeatureHistogram;   //binCount set at compile time
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


    template<class FEATURE_ACCUMULATOR>
    inline void extractBlockFeaturesImpl(const std::string & groupPath,
                                         const std::string & blockPrefix,
                                         const std::string & dataPath,
                                         const std::string & dataKey,
                                         const std::string & labelPath,
                                         const std::string & labelKey,
                                         const std::vector<size_t> & blockIds,
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

            // load the edge-ids
            std::vector<EdgeIndexType> edgeIndices;
            loadEdgeIndices(blockPath.string(), edgeIndices);

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
            accumulator(graph, std::move(data), std::move(labels), roiBegin, roiEnd, edgeIndices);
        }
    }


    ///
    // accumulator implementations
    ///

    // TODO arguments for serializaation
    // accumulate simple boundary map
    inline void accumulateBoundaryMap(const Graph & graph,
                                      std::unique_ptr<z5::Dataset> dataDs,
                                      std::unique_ptr<z5::Dataset> labelsDs,
                                      const std::vector<size_t> & roiBegin,
                                      const std::vector<size_t> & roiEnd,
                                      const std::vector<EdgeIndexType> & edgeIndices,
                                      float dataMin, float dataMax) {
        // xtensor typedegs
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef xt::xtensor<float, 3> DataArray;
        typedef typename DataArray::shape_type ShapeType;

        // get the shapes
        ShapeType shape;
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

        // TODO
        // serialize the accumulators
    }


    // accumulate affinity maps
    inline void accumulateAffinityMap() {
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
                                                     float dataMin=0, float dataMax=1) {

        // TODO could also use the std::bind pattern and std::function
        // TODO capture additional arguments that we need for serialization
        auto accumulator = [dataMin, dataMax](const Graph & graph,
                               std::unique_ptr<z5::Dataset> dataDs,
                               std::unique_ptr<z5::Dataset> labelsDs,
                               const std::vector<size_t> & roiBegin,
                               const std::vector<size_t> & roiEnd,
                               const std::vector<EdgeIndexType> & edgeIndices) {
            accumulateBoundaryMap(graph, std::move(dataDs), std::move(labelsDs),
                                  roiBegin, roiEnd, edgeIndices,
                                  dataMin, dataMax);
        };

        extractBlockFeaturesImpl(groupPath, blockPrefix,
                                 dataPath, dataKey,
                                 labelPath, labelKey,
                                 blockIds, accumulator);
    }


}
}
