#pragma once

#include <unordered_set>
#include <set>
#include <boost/functional/hash.hpp>

#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"

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


    ///
    // graph typedefs at nifty.distributed level
    ///

    typedef uint64_t NodeType;
    typedef int64_t EdgeIndexType;

    typedef std::pair<NodeType, NodeType> EdgeType;
    typedef boost::hash<EdgeType> EdgeHash;

    // Perfoemance for extraction of 50 x 512 x 512 cube (real labels)
    // (including some overhead (python cals, serializing the graph, etc.))
    // using normal set:    1.8720 s
    // using unordered set: 1.8826 s
    // Note that we would need an additional sort to make the unordered set result correct.
    // As we do not see an improvement, stick with the set for now.
    // But for operations on larger edge / node sets, we should benchmark the unordered set again
    typedef std::set<NodeType> NodeSet;
    typedef std::set<EdgeType> EdgeSet;
    //typedef std::unordered_set<EdgeType, EdgeHash> EdgeSet;

    // xtensor typedefs
    typedef xt::xtensor<NodeType, 1> Tensor1;
    typedef xt::xtensor<NodeType, 2> Tensor2;
    typedef typename Tensor1::shape_type Shape1Type;
    typedef typename Tensor2::shape_type Shape2Type;
    typedef xt::xtensor<NodeType, 3> Tensor3;
    typedef typename Tensor3::shape_type Shape3Type;

    // nifty typedef
    typedef nifty::array::StaticArray<int64_t, 3> CoordType;


    ///
    // helper functions (we might turn this into detail namespace ?!)
    ///


    template<class NODES>
    inline void loadNodes(const std::string & graphPath, NODES & nodes) {
        const std::vector<size_t> zero1Coord({0});
        // get handle and dataset
        z5::handle::Group graph(graphPath);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin());
        nodes.insert(tmpNodes.begin(), tmpNodes.end());
    }


    inline void loadNodes(const std::string & graphPath, std::vector<NodeType> & nodes, const size_t offset) {
        const std::vector<size_t> zero1Coord({0});
        // get handle and dataset
        z5::handle::Group graph(graphPath);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin());
        nodes.resize(nodes.size() + nodeShape[0]);
        std::copy(tmpNodes.begin(), tmpNodes.end(), nodes.begin() + offset);
    }


    inline bool loadEdgeIndices(const std::string & graphPath,
                                std::vector<EdgeIndexType> & edgeIndices,
                                const size_t offset) {
        const std::vector<size_t> zero1Coord({0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get id dataset
        auto idDs = z5::openDataset(graph, "edgeIds");
        // read the nodes and inset them into the node set
        Shape1Type idShape({idDs->shape(0)});
        xt::xtensor<EdgeIndexType, 1> tmpIds(idShape);
        z5::multiarray::readSubarray<EdgeIndexType>(idDs, tmpIds, zero1Coord.begin());
        edgeIndices.resize(idShape[0] + edgeIndices.size());
        std::copy(tmpIds.begin(), tmpIds.end(), edgeIndices.begin() + offset);
        return true;
    }


    template<class EDGES>
    inline bool loadEdges(const std::string & graphPath, EDGES & edges) {
        const std::vector<size_t> zero2Coord({0, 0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get edge dataset
        auto edgeDs = z5::openDataset(graph, "edges");
        // read the edges and inset them into the edge set
        Shape2Type edgeShape({edgeDs->shape(0), 2});
        Tensor2 tmpEdges(edgeShape);
        z5::multiarray::readSubarray<NodeType>(edgeDs, tmpEdges, zero2Coord.begin());
        for(size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges.insert(std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1)));
        }
        return true;
    }


    inline bool loadEdges(const std::string & graphPath, std::vector<EdgeType> & edges, const size_t offset) {
        const std::vector<size_t> zero2Coord({0, 0});
        const std::vector<std::string> keys = {"numberOfEdges"};

        // get handle and check if we have edges
        z5::handle::Group graph(graphPath);
        nlohmann::json j;
        z5::readAttributes(graph, keys, j);
        size_t numberOfEdges = j[keys[0]];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get edge dataset
        auto edgeDs = z5::openDataset(graph, "edges");

        // read the edges and inset them into the edge set
        Shape2Type edgeShape({edgeDs->shape(0), 2});
        Tensor2 tmpEdges(edgeShape);
        z5::multiarray::readSubarray<NodeType>(edgeDs, tmpEdges, zero2Coord.begin());
        edges.resize(edges.size() + edgeShape[0]);
        for(size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges[edgeId + offset] = std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1));
        }
        return true;
    }


    // Using templates for node and edge storages here,
    // because we might use different datastructures at different graph levels
    // (set or unordered_set)
    template<class NODES, class EDGES, class COORD>
    void serializeGraph(const std::string & pathToGraph,
                        const std::string & saveKey,
                        const NODES & nodes,
                        const EDGES & edges,
                        const COORD & roiBegin,
                        const COORD & roiEnd,
                        const int numberOfThreads=1,
                        const std::string & compression="raw") {

        const size_t nNodes = nodes.size();
        const size_t nEdges = edges.size();

        // create the graph group
        auto graphPath = fs::path(pathToGraph);
        graphPath /= saveKey;
        z5::handle::Group group(graphPath.string());
        z5::createGroup(group, false);

        // threadpool for parallel writing
        parallel::ThreadPool tp(numberOfThreads);

        // serialize the graph (nodes)
        std::vector<size_t> nodeShape = {nNodes};
        std::vector<size_t> nodeChunks = {std::min(nNodes, 2*262144UL)};
        // FIXME For some reason only raw compression works,
        // because the precompiler flags for activating compression schemes
        // are not properly set (although we can read datasets with compression,
        // so this doesn't make much sense)
        // std::cout << "Writing " << nNodes << " nodes to " << pathToGraph << std::endl;
        auto dsNodes = z5::createDataset(group, "nodes",
                                         "uint64", nodeShape,
                                         nodeChunks, false,
                                         compression);

        const size_t numberNodeChunks = dsNodes->numberOfChunks();
        // std::cout << "Serialize nodes" << std::endl;
        parallel::parallel_foreach(tp, numberNodeChunks, [&](const int tId, const size_t chunkId){
            const size_t nodeStart = chunkId * nodeChunks[0];
            const size_t nodeStop = std::min((chunkId + 1) * nodeChunks[0], nodeShape[0]);

            const size_t nNodesChunk = nodeStop - nodeStart;
            Shape1Type nodeSerShape({nNodesChunk});
            Tensor1 nodeSer(nodeSerShape);

            auto nodeIt = nodes.begin();
            std::advance(nodeIt, nodeStart);
            for(size_t i = 0; i < nNodesChunk; i++, nodeIt++) {
                nodeSer(i) = *nodeIt;
            }

            const std::vector<size_t> nodeOffset({nodeStart});
            z5::multiarray::writeSubarray<NodeType>(dsNodes, nodeSer, nodeOffset.begin());
        });
        // std::cout << "done" << std::endl;

        // serialize the graph (edges)
        // std::cout << "Serialize edges" << std::endl;
        if(nEdges > 0) {
            // std::cout << "Writing " << nEdges << " edges to " << pathToGraph << std::endl;
            std::vector<size_t> edgeShape = {nEdges, 2};
            std::vector<size_t> edgeChunks = {std::min(nEdges, 262144UL), 2};
            // FIXME For some reason only raw compression works,
            // because the precompiler flags for activating compression schemes
            // are not properly set (although we can read datasets with compression,
            // so this doesn't make much sense)
            auto dsEdges = z5::createDataset(group, "edges", "uint64",
                                             edgeShape, edgeChunks, false,
                                             compression);
            const size_t numberEdgeChunks = dsEdges->numberOfChunks();

            parallel::parallel_foreach(tp, numberEdgeChunks, [&](const int tId,
                                                                 const size_t chunkId){
                const size_t edgeStart = chunkId * edgeChunks[0];
                const size_t edgeStop = std::min((chunkId + 1) * edgeChunks[0], edgeShape[0]);

                const size_t nEdgesChunk = edgeStop - edgeStart;
                Shape2Type edgeSerShape({nEdgesChunk, 2});
                Tensor2 edgeSer(edgeSerShape);

                auto edgeIt = edges.begin();
                std::advance(edgeIt, edgeStart);
                for(size_t i = 0; i < nEdgesChunk; i++, edgeIt++) {
                    edgeSer(i, 0) = edgeIt->first;
                    edgeSer(i, 1) = edgeIt->second;
                }

                const std::vector<size_t> edgeOffset({edgeStart, 0});
                z5::multiarray::writeSubarray<NodeType>(dsEdges, edgeSer, edgeOffset.begin());
            });
        }
        // std::cout << "done" << std::endl;

        // serialize metadata (number of edges and nodes and position of the block)
        nlohmann::json attrs;
        attrs["numberOfNodes"] = nNodes;
        attrs["numberOfEdges"] = nEdges;
        attrs["roiBegin"] = std::vector<size_t>(roiBegin.begin(), roiBegin.end());
        attrs["roiEnd"] = std::vector<size_t>(roiEnd.begin(), roiEnd.end());

        z5::writeAttributes(group, attrs);
    }


    inline void makeCoord2(const CoordType & coord, CoordType & coord2, const size_t axis) {
        coord2 = coord;
        coord2[axis] += 1;
    };


    ///
    // Workflow functions
    ///


    template<class COORD>
    void extractGraphFromRoi(const std::string & pathToLabels,
                             const std::string & keyToLabels,
                             const COORD & roiBegin,
                             const COORD & roiEnd,
                             NodeSet & nodes,
                             EdgeSet & edges) {

        // open the n5 label dataset
        auto path = fs::path(pathToLabels);
        path /= keyToLabels;
        auto ds = z5::openDataset(path.string());

        // load the roi
        Shape3Type shape;
        CoordType blockShape, coord2;

        for(int axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - roiBegin[axis];
            blockShape[axis] = shape[axis];
        }
        Tensor3 labels(shape);
        z5::multiarray::readSubarray<NodeType>(ds, labels, roiBegin.begin());

        // iterate over the the roi and extract all graph nodes and edges
        // we want ordered iteration over nodes and edges in the end,
        // so we use a normal set instead of an unordered one

        NodeType lU, lV;
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {

            lU = xtensor::read(labels, coord.asStdArray());
            nodes.insert(lU);
            for(size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);
                if(coord2[axis] < blockShape[axis]){
                    lV = xtensor::read(labels, coord2.asStdArray());
                    if(lU != lV){
                        edges.insert(std::make_pair(std::min(lU, lV), std::max(lU, lV)));
                    }
                }
            }
        });

    }


    template<class COORD>
    inline void computeMergeableRegionGraph(const std::string & pathToLabels,
                                     const std::string & keyToLabels,
                                     const COORD & roiBegin,
                                     const COORD & roiEnd,
                                     const std::string & pathToGraph,
                                     const std::string & keyToRoi) {
        // extract graph nodes and edges from roi
        NodeSet nodes;
        EdgeSet edges;
        extractGraphFromRoi(pathToLabels, keyToLabels,
                            roiBegin, roiEnd,
                            nodes, edges);
        // serialize the graph
        serializeGraph(pathToGraph, keyToRoi,
                       nodes, edges,
                       roiBegin, roiEnd);
    }


    inline void mergeSubgraphsSingleThreaded(const fs::path & graphPath,
                                             const std::string & blockPrefix,
                                             const std::vector<size_t> & blockIds,
                                             NodeSet & nodes,
                                             EdgeSet & edges,
                                             std::vector<size_t> & roiBegin,
                                             std::vector<size_t> & roiEnd) {
        nlohmann::json j;
        const std::vector<std::string> keys({"roiBegin", "roiEnd"});

        fs::path blockPath;
        std::string blockKey;

        for(size_t blockId : blockIds) {

            // open the group associated with the sub-graph corresponding to this block
            blockKey = blockPrefix + std::to_string(blockId);
            blockPath = graphPath;
            blockPath /= blockKey;

            // load nodes and edgees
            loadNodes(blockPath.string(), nodes);
            // std::cout << "Loaded nodes from " << blockId << " now have " << nodes.size() << std::endl;
            loadEdges(blockPath.string(), edges);

            // read the rois from attributes
            z5::handle::Group group(blockPath.string());
            z5::readAttributes(group, keys, j);

            // merge the rois
            const auto & blockBegin = j[keys[0]];
            const auto & blockEnd = j[keys[1]];

            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis], static_cast<size_t>(blockBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis], static_cast<size_t>(blockEnd[axis]));
            }
        }

    }


    inline void mergeSubgraphsMultiThreaded(const fs::path & graphPath,
                                            const std::string & blockPrefix,
                                            const std::vector<size_t> & blockIds,
                                            NodeSet & nodes,
                                            EdgeSet & edges,
                                            std::vector<size_t> & roiBegin,
                                            std::vector<size_t> & roiEnd,
                                            const int numberOfThreads) {
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        auto nThreads = threadpool.nThreads();

        // initialize thread data
        struct PerThreadData {
            std::vector<size_t> roiBegin;
            std::vector<size_t> roiEnd;
            NodeSet nodes;
            EdgeSet edges;
        };
        std::vector<PerThreadData> threadData(nThreads);
        for(int t = 0; t < nThreads; ++t) {
            // FIXME should use some max uint value here
            threadData[t].roiBegin = std::vector<size_t>({10000000, 10000000, 10000000});
            threadData[t].roiEnd = std::vector<size_t>({0, 0, 0});
        }

        // merge nodes and edges multi threaded
        size_t nBlocks = blockIds.size();
        const std::vector<std::string> keys({"roiBegin", "roiEnd"});
        // std::cout << "Merging subgraphs ..." << std::endl;
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int tid, const int blockIndex){

            // get the thread data
            auto blockId = blockIds[blockIndex];
            auto * threadNodes = &threadData[tid].nodes;
            auto * threadEdges = &threadData[tid].edges;
            // for thread 0, we use the input sets instead of our thread data
            // to avoid one sequential merge in the end
            if(tid == 0) {
                threadNodes = &nodes;
                threadEdges = &edges;
            }
            auto & threadBegin = threadData[tid].roiBegin;
            auto & threadEnd = threadData[tid].roiEnd;

            // open the group associated with the sub-graph corresponding to this block
            std::string blockKey = blockPrefix + std::to_string(blockId);
            fs::path blockPath = graphPath;
            blockPath /= blockKey;

            // load nodes and edgees
            loadNodes(blockPath.string(), *threadNodes);
            loadEdges(blockPath.string(), *threadEdges);

            // read the rois from attributes
            nlohmann::json j;
            z5::handle::Group group(blockPath.string());
            z5::readAttributes(group, keys, j);

            // merge the rois
            const auto & blockBegin = j[keys[0]];
            const auto & blockEnd = j[keys[1]];

            for(int axis = 0; axis < 3; ++axis) {
                threadBegin[axis] = std::min(threadBegin[axis], static_cast<size_t>(blockBegin[axis]));
                threadEnd[axis] = std::max(threadEnd[axis], static_cast<size_t>(blockEnd[axis]));
            }
        });

        // merge into final nodes and edges
        // (note that thread 0 was already used for the input nodes and edges)
        for(int tid = 1; tid < nThreads; ++tid) {
            nodes.insert(threadData[tid].nodes.begin(), threadData[tid].nodes.end());
            edges.insert(threadData[tid].edges.begin(), threadData[tid].edges.end());
        }

        // merge the rois
        for(int tid = 0; tid < nThreads; ++tid) {
            const auto & threadBegin = threadData[tid].roiBegin;
            const auto & threadEnd = threadData[tid].roiEnd;
            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis], static_cast<size_t>(threadBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis], static_cast<size_t>(threadEnd[axis]));
            }
        }
        // std::cout << "done" << std::endl;
    }


    inline void mergeSubgraphs(const std::string & pathToGraph,
                               const std::string & blockPrefix,
                               const std::vector<size_t> & blockIds,
                               const std::string & outKey,
                               const int numberOfThreads=1) {
        // TODO we should try unordered sets again here
        NodeSet nodes;
        EdgeSet edges;

        // FIXME should use some max uint value here
        std::vector<size_t> roiBegin({10000000, 10000000, 10000000});
        std::vector<size_t> roiEnd({0, 0, 0});

        if(numberOfThreads == 1) {
            mergeSubgraphsSingleThreaded(pathToGraph, blockPrefix, blockIds,
                                         nodes, edges,
                                         roiBegin, roiEnd);
        } else {
            mergeSubgraphsMultiThreaded(pathToGraph, blockPrefix,
                                        blockIds, nodes, edges,
                                        roiBegin, roiEnd,
                                        numberOfThreads);
        }

        // we can only use compression for
        // big enough blocks (too small chunks will result in zlib error)
        // as a proxy we use the number of threads to determine if we use compression
        std::string compression = (numberOfThreads > 1) ? "gzip" : "raw";
        // serialize the merged graph
        serializeGraph(pathToGraph, outKey,
                       nodes, edges,
                       roiBegin, roiEnd,
                       numberOfThreads,
                       compression);
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                    const std::string & graphKey,
                    const std::string & blockPrefix,
                    const std::vector<size_t> & blockIds,
                    const int numberOfThreads=1) {

        const std::vector<size_t> zero1Coord({0});
        // we load the edges into a vector, because
        // it will be sorted by construction and we can take
        // advantage of O(logN) search with std::lower_bound
        std::vector<EdgeType> edges;
        fs::path graphPath(pathToGraph);
        graphPath /= graphKey;
        loadEdges(graphPath.string(), edges, 0);

        // iterate over the blocks and insert the nodes and edges
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        auto nThreads = threadpool.nThreads();
        size_t nBlocks = blockIds.size();

        // handle all the blocks in parallel
        nifty::parallel::parallel_foreach(threadpool, nBlocks, [&](const int tid, const int blockIndex){

            auto blockId = blockIds[blockIndex];

            // open the group associated with the sub-graph corresponding to this block
            const std::string blockKey = blockPrefix + std::to_string(blockId);
            fs::path blockPath(pathToGraph);
            blockPath /= blockKey;

            // load the block edges
            std::vector<EdgeType> blockEdges;
            bool haveEdges = loadEdges(blockPath.string(), blockEdges, 0);
            if(!haveEdges) {
                return;
            }

            // label the local edges acccording to the global edge ids
            std::vector<EdgeIndexType> edgeIds(blockEdges.size());

            // find the first local edge in the global edges
            auto edgeIt = std::lower_bound(edges.begin(), edges.end(), blockEdges[0]);

            // it is guaranteed that all local edges are 'above' the lowest we just found,
            // hence we start searching from this edge, and always try to increase the
            // edge iterator by one before searching again, because edges are likely to be close spatially
            for(EdgeIndexType localEdgeId = 0; localEdgeId < blockEdges.size(); ++localEdgeId) {
                const EdgeType & edge = *edgeIt;
                if(blockEdges[localEdgeId] == edge) {
                    edgeIds[localEdgeId] = std::distance(edges.begin(), edgeIt);
                    ++edgeIt;
                } else {
                    edgeIt = std::lower_bound(edgeIt, edges.end(), blockEdges[localEdgeId]);
                    edgeIds[localEdgeId] = std::distance(edges.begin(), edgeIt);
                }
            }

            // serialize the edge ids
            std::vector<size_t> idShape = {edgeIds.size()};
            auto idView = xt::adapt(edgeIds, idShape);
            z5::handle::Group block(blockPath.string());
            auto dsIds = z5::createDataset(block, "edgeIds", "int64", idShape, idShape, false);
            z5::multiarray::writeSubarray<EdgeIndexType>(dsIds, idView, zero1Coord.begin());
        });
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                    const std::string & graphKey,
                    const std::string & blockPrefix,
                    const size_t numberOfBlocks,
                    const int numberOfThreads=1) {
        std::vector<size_t> blockIds(numberOfBlocks);
        std::iota(blockIds.begin(), blockIds.end(), 0);
        mapEdgeIds(pathToGraph, graphKey, blockPrefix, blockIds, numberOfThreads);
    }

}
}
