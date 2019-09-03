#pragma once

#include "nifty/tools/blocking.hxx"
#include "nifty/distributed/mergeable_features.hxx"

#ifdef WITH_BOOST_FS
    namespace fs = boost::filesystem;
#else
    #if __GCC__ > 7
        namespace fs = std::filesystem;
    #else
        namespace fs = std::experimental::filesystem;
    #endif
#endif


namespace nifty {
namespace distributed {


    template<class COORD>
    inline void increaseRoi(COORD & roiBegin) {
        // we increase the roi by decreasing roiBegin by 1.
        // to match what was done in the graph extraction when increaseRoi is true
        for(int axis = 0; axis < 3; ++axis) {
            if(roiBegin[axis] > 0) {
                --roiBegin[axis];
            }
        }

    }


    template<class COORD>
    inline void loadBB(const COORD & begIn, const COORD & endIn,
                       std::vector<std::size_t> & begOut, std::vector<std::size_t> & endOut) {
        std::copy(begIn.begin(), begIn.end(), begOut.begin());
        std::copy(endIn.begin(), endIn.end(), endOut.begin());
        increaseRoi(begOut);
    }


    template<class GRAPH, class LABELS, class OUT>
    inline void find1DEdgesBlock(const GRAPH & graph, const LABELS & labels,
                                 const std::vector<EdgeIndexType> & edgeIndices,
                                 const bool ignoreLabel, OUT & out, std::vector<uint8_t> & edgeAxes) {
        typedef nifty::array::StaticArray<int64_t, 3> CoordType;
        CoordType shape;
        std::copy(labels.shape().begin(), labels.shape().end(), shape.begin());

        nifty::tools::forEachCoordinate(shape, [&](const CoordType & coord) {
            const NodeType lU = xtensor::read(labels, coord.asStdArray());

            // check for ignore label
            if(lU == 0 && ignoreLabel) {
                return;
            }

            CoordType coord2;
            for(std::size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);

                // check if this is an actual edge
                if(coord2[axis] >= shape[axis]){
                    continue;
                }
                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lV == 0 && ignoreLabel) {
                    continue;
                }
                if(lU == lV){
                    continue;
                }

                // yes it is!
                const auto edgeId = graph.findEdge(lU, lV);
                const auto globalEdgeId = edgeIndices[edgeId];

                // did we already visit the edge - and if so, what's the prev state?
                auto & prevState = out(globalEdgeId);
                // prev state is 0 -> the edge was not visited yet -> we just set our current axis
                if(prevState == 0) {
                    edgeAxes[globalEdgeId] = axis;
                    prevState = 1;  // set prev state to visited and 1d axis
                } else if(prevState == 1) {  // prev state is 1 -> edge was visited and is still 1d axes
                    // if the previous axis does not correspond to the current axis, set
                    // edge status to 2 = mixed axes
                    if(edgeAxes[globalEdgeId] != axis) {
                        prevState = 2;
                    }
                }
                // otherwise prevState is 2 -> edge is already mixed axes
            }
        });
    }


    template<class OUT>
    inline void find1DEdges(const std::string & graphPath,
                            const std::string & graphPrefix,
                            const std::string & labelPath,
                            const std::string & labelKey,
                            const std::vector<std::size_t> & blockIds,
                            OUT & out) {
        typedef xt::xtensor<NodeType, 3> LabelArray;
        std::vector<uint8_t> edgeAxes(out.size());

        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::File labelFile(labelPath);
        auto labelDs = z5::openDataset(labelFile, labelKey);

        std::vector<std::size_t> roiBegin(3), roiEnd(3);
        for(const std::size_t blockId : blockIds) {

            // load the sub-graph corresponding to this block
            const std::string blockKey = graphPrefix + std::to_string(blockId);
            Graph graph(graphPath, blockKey);
            // continue if we don't have edges in this graph
            if(graph.numberOfEdges() == 0) {
                continue;
            }

            // load the bounding box information
            z5::filesystem::handle::Group group(graphFile, blockKey);
            nlohmann::json j;
            z5::readAttributes(group, j);
            loadBB(j["roiBegin"], j["roiEnd"], roiBegin, roiEnd);

            // get the shape and create array
            Shape3Type shape;
            for(unsigned axis = 0; axis < 3; ++axis) {
                shape[axis] = roiEnd[axis] - roiBegin[axis];
            }
            LabelArray labels(shape);

            // load the labels from the bounding box
            z5::multiarray::readSubarray<NodeType>(labelDs, labels, roiBegin.begin());
            const bool ignoreLabel = j["ignoreLabel"];

            // load the global edge indices for this block
            std::vector<EdgeIndexType> edgeIndices;
            loadEdgeIndices(group, edgeIndices, 0);

            // find which edges are 1d
            find1DEdgesBlock(graph, labels, edgeIndices, ignoreLabel, out, edgeAxes);
        }
    }


    template<class GRAPH, class LABELS, class BLOCKING, class OUT>
    inline void findBlockBoundaryEdgesBlock(const GRAPH & graph,
                                            const LABELS & labels,
                                            const std::vector<EdgeIndexType> & edgeIndices,
                                            const BLOCKING & blocking,
                                            const std::size_t blockId,
                                            const bool ignoreLabel,
                                            OUT & out) {


        typedef nifty::array::StaticArray<int64_t, 3> CoordType;
        CoordType shape;
        std::copy(labels.shape().begin(), labels.shape().end(), shape.begin());

        // iterate over the block faces
        for(unsigned axis = 0; axis < 3; ++axis) {
            // check if this is a border block == has no lower neighbor in this axis
            if(blocking.getNeighborId(blockId, axis, true) == -1) {
                continue;
            }

            // get the shape of this face
            CoordType faceShape;
            for(unsigned d = 0; d < 3; ++d) {
                faceShape[d] = (d == axis) ? 1 : shape[d];
            }

            nifty::tools::forEachCoordinate(faceShape, [&](const CoordType & coord) {
                const NodeType lU = xtensor::read(labels, coord.asStdArray());
                // check for ignore label
                if(lU == 0 && ignoreLabel) {
                    return;
                }

                CoordType coord2 = coord;
                coord2[axis] = 1;

                const NodeType lV = xtensor::read(labels, coord2.asStdArray());
                if(lV == 0 && ignoreLabel) {
                    return;
                }
                if(lU == lV){
                    return;
                }

                const auto edgeId = graph.findEdge(lU, lV);
                const auto globalEdgeId = edgeIndices[edgeId];
                out(globalEdgeId) = true;
            });
        }
    }


    template<class OUT>
    inline void findBlockBoundaryEdges(const std::string & graphPath,
                                       const std::string & graphPrefix,
                                       const std::string & labelPath,
                                       const std::string & labelKey,
                                       const std::vector<std::size_t> & blockShape,
                                       const std::vector<std::size_t> & blockIds,
                                       OUT & out) {
        typedef xt::xtensor<NodeType, 3> LabelArray;
        typedef nifty::array::StaticArray<int64_t, 3> VectorType;

        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::File labelFile(labelPath);
        auto labelDs = z5::openDataset(labelFile, labelKey);

        VectorType shape, blockShapeVec;
        std::copy(labelDs->shape().begin(), labelDs->shape().end(), shape.begin());
        std::copy(blockShape.begin(), blockShape.end(), blockShapeVec.begin());

        tools::Blocking<3> blocking(VectorType({0, 0, 0}),
                                    shape, blockShapeVec);

        for(const std::size_t blockId : blockIds) {

            // load the sub-graph corresponding to this block
            const std::string blockKey = graphPrefix + std::to_string(blockId);
            Graph graph(graphPath, blockKey);
            // continue if we don't have edges in this graph
            if(graph.numberOfEdges() == 0) {
                continue;
            }

            const auto block = blocking.getBlock(blockId);
            auto roiBegin = block.begin();
            const auto & roiEnd = block.end();
            increaseRoi(roiBegin);

            // load the bounding box information
            z5::filesystem::handle::Group group(graphFile, blockKey);
            nlohmann::json j;
            z5::readAttributes(group, j);

            // get the shape and create array
            Shape3Type thisShape;
            for(unsigned axis = 0; axis < 3; ++axis) {
                thisShape[axis] = roiEnd[axis] - roiBegin[axis];
            }
            LabelArray labels(thisShape);

            // load the labels from the bounding box
            z5::multiarray::readSubarray<NodeType>(labelDs, labels, roiBegin.begin());
            const bool ignoreLabel = j["ignoreLabel"];

            // load the global edge indices for this block
            std::vector<EdgeIndexType> edgeIndices;
            loadEdgeIndices(group, edgeIndices, 0);

            // find which edges are 1d
            findBlockBoundaryEdgesBlock(graph, labels, edgeIndices, blocking, blockId,
                                        ignoreLabel, out);
        }
    }


}
}
