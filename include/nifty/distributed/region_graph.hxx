#pragma once

#include <unordered_set>
#include <set>

#include "xtensor/xtensor.hpp"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/dataset_factory.hxx"
#include "z5/groups.hxx"
#include "z5/attributes.hxx"

#include "nifty/array/static_array.hxx"
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {

    template<class COORD>
    void computeMergeableRegionGraph(const std::string & pathToLabels,
                                     const std::string & keyToLabels,
                                     const COORD & roiBegin,
                                     const COORD & roiEnd,
                                     const std::string & pathToGraph,
                                     const std::string & keyToRoi) {
        // typedefs
        typedef xt::xtensor<uint64_t, 1> Tensor1;
        typedef xt::xtensor<uint64_t, 2> Tensor2;
        typedef xt::xtensor<uint64_t, 3> Tensor3;
        typedef typename Tensor1::shape_type Shape1Type;
        typedef typename Tensor2::shape_type Shape2Type;
        typedef typename Tensor3::shape_type Shape3Type;
        typedef nifty::array::StaticArray<int64_t, 3> CoordType;

        std::vector<size_t> zero1Coord({0});
        std::vector<size_t> zero2Coord({0, 0});


        // open the n5 label dataset
        auto path = fs::path(pathToLabels);
        path /= keyToLabels;
        auto ds = z5::openDataset(path.string());


        // load the roi
        Shape3Type shape;
        CoordType blockShape;

        for(int axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - roiBegin[axis];
            blockShape[axis] = shape[axis];
        }
        Tensor3 labels(shape);
        z5::multiarray::readSubarray<uint64_t>(ds, labels, roiBegin.begin());

        // iterate over the the roi and extract all graph nodes and edges
        // we want ordered iteration over nodes and edges in the end,
        // so we use a normal set instead of an unordered one
        // TODO don't really know, maybe we should still use unordered sets
        // and just copy to normal set or vector for sorted iteration
        std::set<uint64_t> nodes;
        std::set<std::pair<uint64_t, uint64_t>> edges;

        auto makeCoord2 = [](const CoordType & coord, const size_t axis){
            CoordType coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        uint64_t lU, lV;
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {

            lU = xtensor::read(labels, coord.asStdArray());
            nodes.insert(lV);
            for(size_t axis = 0; axis < 3; ++axis){
                const auto coord2 = makeCoord2(coord, axis);
                if(coord2[axis] < blockShape[axis]){
                    lV = xtensor::read(labels, coord2.asStdArray());
                    if(lU != lV){
                        edges.insert(std::make_pair(std::min(lU, lV), std::max(lU, lV)));
                    }
                }
            }
        });

        size_t nNodes = nodes.size();
        size_t nEdges = edges.size();


        // create the graph group
        auto graphPath = fs::path(pathToGraph);
        graphPath /= keyToRoi;
        z5::handle::Group group(graphPath.string());
        z5::createGroup(group, false);


        // serialize the graph (edges and nodes)
        // TODO should we additionally chunk / compress this ?
        std::vector<size_t> nodeShape({nNodes});
        auto dsNodes = z5::createDataset(group, "nodes", "uint64", nodeShape, nodeShape, false);
        Shape1Type nodeSerShape({nNodes});
        Tensor1 nodeSer(nodeSerShape);
        size_t i = 0;
        for(const auto node : nodes) {
            nodeSer[i] = node;
            ++i;
        }
        z5::multiarray::writeSubarray<uint64_t>(dsNodes, nodeSer, zero1Coord.begin());

        std::vector<size_t> edgeShape({nEdges, 2});
        auto dsEdges = z5::createDataset(group, "edges", "uint64", edgeShape, edgeShape, false);
        Shape2Type edgeSerShape({nEdges, 2});
        Tensor2 edgeSer(edgeSerShape);
        i = 0;
        for(const auto & edge : edges) {
            edgeSer[i, 0] = edge.first;
            edgeSer[i, 1] = edge.second;
            ++i;
        }
        z5::multiarray::writeSubarray<uint64_t>(dsEdges, edgeSer, zero2Coord.begin());


        // serialize metadata (number of edges and nodes and position of the block)
        nlohmann::json attrs;
        attrs["numberOfNodes"] = nNodes;
        attrs["numberOfEdges"] = nEdges;
        attrs["roiBegin"] = std::vector<size_t>(roiBegin.begin(), roiBegin.end());
        attrs["roiEnd"] = std::vector<size_t>(roiEnd.begin(), roiEnd.end());

        z5::writeAttributes(group, attrs);
    }


}
}
