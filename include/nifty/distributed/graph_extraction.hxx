#pragma once

#include <unordered_set>
#include <set>
#include <boost/functional/hash.hpp>

#include "xtensor/xtensor.hpp"
#include "xtensor/xadapt.hpp"

#include "z5/factory.hxx"
#include "z5/attributes.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "nifty/array/static_array.hxx"
#include "nifty/xtensor/xtensor.hxx"
#include "nifty/tools/for_each_coordinate.hxx"


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


    // load labels from block; works for normal label dataset
    // and label multisets
    template<class LABELS, class ROI>
    inline bool loadLabels(const std::string & path, const std::string & key,
                           LABELS & labels, const ROI & roiBegin) {

        typedef typename LABELS::value_type NodeType;

        z5::filesystem::handle::File file(path);
        auto ds = z5::openDataset(file, key);

        // check if this is a label multiset
        bool isLabelMultiset = false;

        // TODO check for label multiset attributes

        if(isLabelMultiset) {
            // is a label multiset -> load the argmax vector and copy into label array

            // first, need to find this chunk id
            const auto & chunks = ds->defaultChunkShape();
            const unsigned ndim = chunks.size();
            std::vector<std::size_t> chunkId(ndim);
            for(unsigned d = 0; d < ndim; ++d) {
                chunkId[d] = roiBegin[d] / chunks[d];
            }

            // next, load this chunk, if we don't have one, return false

            // find the number of labels in this chunk
            // (encoded in the first 4 bytes as signed integer '>i' in pythons struct)

            // load the first 8 * number of labels bytes, which encode the argmax labels
            // encoded as '>q' in python's struct TODO unsigned long?

            // reshape the label vector and copy it into the label output array
        }
        else {
            // not a label multiset -> we can just load the label array
            z5::multiarray::readSubarray<NodeType>(ds, labels, roiBegin.begin());

            // check if the labels are all zero
        }
        return true;
    }


    template<class HANDLE, class NODES>
    inline void loadNodes(const HANDLE & graph, NODES & nodes) {
        const std::vector<std::size_t> zero1Coord({0});
        // get dataset
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin());
        nodes.insert(tmpNodes.begin(), tmpNodes.end());
    }

    template<class NODES>
    inline void loadNodes(const std::string & graphPath,
                          const std::string & graphKey,
                          NODES & nodes) {
        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::Group graph(graphFile, graphKey);
        loadNodes(graph, nodes);
    }


    template<class HANDLE>
    inline void loadNodes(const HANDLE & graph,
                          std::vector<NodeType> & nodes,
                          const std::size_t offset,
                          const int nThreads=1) {
        const std::vector<std::size_t> zero1Coord({0});
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the node set
        Shape1Type nodeShape({nodeDs->shape(0)});
        Tensor1 tmpNodes(nodeShape);
        z5::multiarray::readSubarray<NodeType>(nodeDs, tmpNodes, zero1Coord.begin(), nThreads);
        nodes.resize(nodes.size() + nodeShape[0]);
        std::copy(tmpNodes.begin(), tmpNodes.end(), nodes.begin() + offset);

    }

    inline void loadNodes(const std::string & graphPath,
                          const std::string & graphKey,
                          std::vector<NodeType> & nodes,
                          const std::size_t offset,
                          const int nThreads=1) {
        // get handle and dataset
        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::Group graph(graphFile, graphKey);
        loadNodes(graph, nodes, offset, nThreads);
    }


    template<class NODES>
    inline void loadNodesToArray(const std::string & graphPath,
                                 const std::string & graphKey,
                                 xt::xexpression<NODES> & nodesExp,
                                 const int nThreads=1) {
        auto & nodes = nodesExp.derived_cast();
        const std::vector<std::size_t> zero1Coord({0});
        // get handle and dataset
        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::Group graph(graphFile, graphKey);
        auto nodeDs = z5::openDataset(graph, "nodes");
        // read the nodes and inset them into the array
        z5::multiarray::readSubarray<NodeType>(nodeDs, nodes, zero1Coord.begin(), nThreads);
    }


    template<class HANDLE>
    inline bool loadEdgeIndices(const HANDLE & graph,
                                std::vector<EdgeIndexType> & edgeIndices,
                                const std::size_t offset,
                                const int nThreads=1) {
        const std::vector<std::size_t> zero1Coord({0});
        // check if we have edges
        nlohmann::json j;
        z5::readAttributes(graph, j);
        std::size_t numberOfEdges = j["numberOfEdges"];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get id dataset
        auto idDs = z5::openDataset(graph, "edgeIds");
        // read the nodes and inset them into the node set
        Shape1Type idShape({idDs->shape(0)});
        xt::xtensor<EdgeIndexType, 1> tmpIds(idShape);
        z5::multiarray::readSubarray<EdgeIndexType>(idDs, tmpIds, zero1Coord.begin(), nThreads);
        edgeIndices.resize(idShape[0] + edgeIndices.size());
        std::copy(tmpIds.begin(), tmpIds.end(), edgeIndices.begin() + offset);
        return true;
    }


    inline bool loadEdgeIndices(const std::string & graphPath,
                                const std::string & graphKey,
                                std::vector<EdgeIndexType> & edgeIndices,
                                const std::size_t offset,
                                const int nThreads=1) {
        const z5::filesystem::handle::File graphFile(graphPath);
        const z5::filesystem::handle::Group graph(graphFile, graphKey);
        loadEdgeIndices(graph, edgeIndices, offset, nThreads);
    }


    template<class HANDLE, class EDGES>
    inline bool loadEdges(const HANDLE & graph, EDGES & edges) {
        const std::vector<std::size_t> zero2Coord({0, 0});
        // check if we have edges
        nlohmann::json j;
        z5::readAttributes(graph, j);
        std::size_t numberOfEdges = j["numberOfEdges"];

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
        for(std::size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges.insert(std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1)));
        }
        return true;
    }

    template<class EDGES>
    inline bool loadEdges(const std::string & graphPath,
                          const std::string & graphKey,
                          EDGES & edges) {
        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::Group graph(graphFile, graphKey);
        loadEdges(graph, edges);
    }


    template<class HANDLE>
    inline bool loadEdges(const HANDLE & graph,
                          std::vector<EdgeType> & edges,
                          const std::size_t offset,
                          const int nThreads=1) {
        const std::vector<std::size_t> zero2Coord({0, 0});
        // check if we have edges
        nlohmann::json j;
        z5::readAttributes(graph, j);
        std::size_t numberOfEdges = j["numberOfEdges"];

        // don't do anything, if we don't have edges
        if(numberOfEdges == 0) {
            return false;
        }

        // get edge dataset
        auto edgeDs = z5::openDataset(graph, "edges");

        // read the edges and inset them into the edge set
        Shape2Type edgeShape({edgeDs->shape(0), 2});
        Tensor2 tmpEdges(edgeShape);
        z5::multiarray::readSubarray<NodeType>(edgeDs, tmpEdges, zero2Coord.begin(), nThreads);
        edges.resize(edges.size() + edgeShape[0]);
        for(std::size_t edgeId = 0; edgeId < edgeShape[0]; ++edgeId) {
            edges[edgeId + offset] = std::make_pair(tmpEdges(edgeId, 0), tmpEdges(edgeId, 1));
        }
        return true;
    }


    inline bool loadEdges(const std::string & graphPath,
                          const std::string & graphKey,
                          std::vector<EdgeType> & edges,
                          const std::size_t offset,
                          const int nThreads=1) {
        z5::filesystem::handle::File graphFile(graphPath);
        z5::filesystem::handle::Group graph(graphFile, graphKey);
        loadEdges(graph, edges, offset, nThreads);
    }


    // Using templates for node and edge storages here,
    // because we might use different datastructures at different graph levels
    // (set or unordered_set)
    template<class NODES, class EDGES, class COORD>
    void serializeGraph(const std::string & pathToGraph,
                        const std::string & graphKey,
                        const NODES & nodes,
                        const EDGES & edges,
                        const COORD & roiBegin,
                        const COORD & roiEnd,
                        const int numberOfThreads=1,
                        const std::string & compression="gzip") {

        const std::size_t nNodes = nodes.size();
        const std::size_t nEdges = edges.size();

        // create the graph group
        z5::filesystem::handle::File graphFile(pathToGraph);
        if(!graphFile.in(graphKey)) {
            z5::createGroup(graphFile, graphKey);
        }
        z5::filesystem::handle::Group group(graphFile, graphKey);

        // threadpool for parallel writing
        parallel::ThreadPool tp(numberOfThreads);

        // serialize the graph (nodes)
        std::vector<std::size_t> nodeShape = {nNodes};
        std::vector<std::size_t> nodeChunks = {std::min(nNodes, 2*262144UL)};
        auto dsNodes = z5::createDataset(group, "nodes", "uint64", nodeShape,
                                         nodeChunks, compression);

        const std::size_t numberNodeChunks = dsNodes->numberOfChunks();
        parallel::parallel_foreach(tp, numberNodeChunks, [&](const int tId,
                                                             const std::size_t chunkId){
            const std::size_t nodeStart = chunkId * nodeChunks[0];
            const std::size_t nodeStop = std::min((chunkId + 1) * nodeChunks[0],
                                             nodeShape[0]);

            const std::size_t nNodesChunk = nodeStop - nodeStart;
            Shape1Type nodeSerShape({nNodesChunk});
            Tensor1 nodeSer(nodeSerShape);

            auto nodeIt = nodes.begin();
            std::advance(nodeIt, nodeStart);
            for(std::size_t i = 0; i < nNodesChunk; i++, nodeIt++) {
                nodeSer(i) = *nodeIt;
            }

            const std::vector<std::size_t> nodeOffset({nodeStart});
            z5::multiarray::writeSubarray<NodeType>(dsNodes, nodeSer,
                                                    nodeOffset.begin());
        });

        // serialize the graph (edges)
        if(nEdges > 0) {
            std::vector<std::size_t> edgeShape = {nEdges, 2};
            std::vector<std::size_t> edgeChunks = {std::min(nEdges, 262144UL), 2};

            auto dsEdges = z5::createDataset(group, "edges", "uint64",
                                             edgeShape, edgeChunks, compression);
            const std::size_t numberEdgeChunks = dsEdges->numberOfChunks();

            parallel::parallel_foreach(tp, numberEdgeChunks, [&](const int tId,
                                                                 const std::size_t chunkId){
                const std::size_t edgeStart = chunkId * edgeChunks[0];
                const std::size_t edgeStop = std::min((chunkId + 1) * edgeChunks[0],
                                                 edgeShape[0]);

                const std::size_t nEdgesChunk = edgeStop - edgeStart;
                Shape2Type edgeSerShape({nEdgesChunk, 2});
                Tensor2 edgeSer(edgeSerShape);

                auto edgeIt = edges.begin();
                std::advance(edgeIt, edgeStart);
                for(std::size_t i = 0; i < nEdgesChunk; i++, edgeIt++) {
                    edgeSer(i, 0) = edgeIt->first;
                    edgeSer(i, 1) = edgeIt->second;
                }

                const std::vector<std::size_t> edgeOffset({edgeStart, 0});
                z5::multiarray::writeSubarray<NodeType>(dsEdges, edgeSer,
                                                        edgeOffset.begin());
            });
        }

        // serialize metadata (number of edges and nodes and position of the block)
        nlohmann::json attrs;
        attrs["numberOfNodes"] = nNodes;
        attrs["numberOfEdges"] = nEdges;
        attrs["roiBegin"] = std::vector<std::size_t>(roiBegin.begin(), roiBegin.end());
        attrs["roiEnd"] = std::vector<std::size_t>(roiEnd.begin(), roiEnd.end());

        z5::writeAttributes(group, attrs);
    }


    // Using templates for node and edge storages here,
    // because we might use different datastructures at different graph levels
    // (set or unordered_set)
    template<class NODES, class EDGES, class COORD>
    void serializeGraphToVarlen(const std::string & pathToGraph,
                                const std::string & graphKey,
                                const NODES & nodes,
                                const EDGES & edges,
                                const COORD & roiBegin) {

        const std::size_t nNodes = nodes.size();
        const std::size_t nEdges = edges.size();

        // no nodes -> we do nothing
        if(nNodes == 0) {
            return;
        }

        z5::filesystem::handle::File graphFile(pathToGraph);

        // get the node varlen dataset
        const std::string nodeKey = graphKey + "/nodes";
        const auto dsNodes = z5::openDataset(graphFile, nodeKey);

        // get the current chunk id
        const auto & chunks = dsNodes->defaultChunkShape();
        const unsigned ndim = chunks.size();
        std::vector<std::size_t> chunkId(roiBegin.begin(), roiBegin.end());
        for(unsigned d = 0; d < ndim; ++d) {
            chunkId[d] /= chunks[d];
        }

        // write the node chunk as varlen
        std::vector<uint64_t> nodeSer(nodes.begin(), nodes.end());
        dsNodes->writeChunk(chunkId, &nodeSer[0], true, nNodes);

        // no edges -> return
        if(nEdges == 0) {
            return;
        }

        // get the edge varlen dataset
        const std::string edgeKey = graphKey + "/edges";
        const auto dsEdges = z5::openDataset(graphFile, edgeKey);

        // write the flattened edge chunk as varlen
        std::vector<uint64_t> edgeSer(2 * nEdges);
        std::size_t flatId = 0;
        for(const auto & edge : edges) {
            edgeSer[flatId] = edge.first;
            ++flatId;
            edgeSer[flatId] = edge.second;
            ++flatId;
        }
        dsEdges->writeChunk(chunkId, &edgeSer[0], true, 2 * nEdges);
    }


    inline void makeCoord2(const CoordType & coord,
                           CoordType & coord2,
                           const std::size_t axis) {
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
                             EdgeSet & edges,
                             const bool ignoreLabel=false,
                             const bool increaseRoi=true) {

        // if specified, we decrease roiBegin by 1.
        // this is necessary to capture edges that lie in between of block boundaries
        // However, we don't want to add the nodes to nodes in the sub-graph !
        COORD actualRoiBegin = roiBegin;
        std::array<bool, 3> roiIncreasedAxis = {false, false, false};
        if(increaseRoi) {
            for(int axis = 0; axis < 3; ++axis) {
                if(actualRoiBegin[axis] > 0) {
                    --actualRoiBegin[axis];
                    roiIncreasedAxis[axis] = true;
                }
            }
        }

        // load the roi
        Shape3Type shape;
        CoordType blockShape, coord2;

        for(int axis = 0; axis < 3; ++axis) {
            shape[axis] = roiEnd[axis] - actualRoiBegin[axis];
            blockShape[axis] = shape[axis];
        }
        Tensor3 labels(shape);

        // load the label block from n5
        const bool hasLabels = loadLabels(pathToLabels, keyToLabels, labels, actualRoiBegin);

        // don't write empty labels
        if(!hasLabels) {
            return;
        }

        // iterate over the the roi and extract all graph nodes and edges
        // we want ordered iteration over nodes and edges in the end,
        // so we use a normal set instead of an unordered one

        NodeType lU, lV;
        nifty::tools::forEachCoordinate(blockShape,[&](const CoordType & coord) {

            lU = xtensor::read(labels, coord.asStdArray());
            // we don't add the nodes in the increased roi
            if(increaseRoi) {
                bool insertNode = true;
                for(int axis = 0; axis < 3; ++axis) {
                    if(coord[axis] == 0 && roiIncreasedAxis[axis]) {
                        insertNode = false;
                        break;
                    }
                }
                if(insertNode) {
                    nodes.insert(lU);
                }
            }
            else {
                nodes.insert(lU);
            }

            // skip edges to zero if we have an ignoreLabel
            if(ignoreLabel && (lU == 0)) {
                return;
            }

            for(std::size_t axis = 0; axis < 3; ++axis){
                makeCoord2(coord, coord2, axis);
                if(coord2[axis] < blockShape[axis]){
                    lV = xtensor::read(labels, coord2.asStdArray());
                    // skip zero if we have an ignoreLabel
                    if(ignoreLabel && (lV == 0)) {
                        continue;
                    }
                    if(lU != lV){
                        edges.insert(std::make_pair(std::min(lU, lV),
                                     std::max(lU, lV)));
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
                                            const std::string & keyToRoi,
                                            const bool ignoreLabel=false,
                                            const bool increaseRoi=false,
                                            const bool serializeToVarlen=false) {
        // extract graph nodes and edges from roi
        NodeSet nodes;
        EdgeSet edges;
        extractGraphFromRoi(pathToLabels, keyToLabels,
                            roiBegin, roiEnd,
                            nodes, edges,
                            ignoreLabel, increaseRoi);
        // serialize the graph
        if(serializeToVarlen) {
            serializeGraphToVarlen(pathToGraph, keyToRoi,
                                   nodes, edges,
                                   roiBegin);
        } else {
            serializeGraph(pathToGraph, keyToRoi,
                           nodes, edges,
                           roiBegin, roiEnd);
        }
    }


    inline void mergeSubgraphsSingleThreaded(const std::string & graphPath,
                                             const std::string & subgraphKey,
                                             const std::vector<std::size_t> & chunkIds,
                                             NodeSet & nodes,
                                             EdgeSet & edges,
                                             std::vector<std::size_t> & roiBegin,
                                             std::vector<std::size_t> & roiEnd) {
        // open the graph file
        z5::filesystem::handle::File graph(graphPath);

        // open the node and edge varlen dataset
        const std::string nodeKey = subgraphKey + "/nodes";
        const auto dsNodes = z5::openDataset(graph, nodeKey);
        const std::string edgeKey = subgraphKey + "/edges";
        const auto dsEdges = z5::openDataset(graph, edgeKey);

        // get blocking of the dataset
        const auto & blocking = dsNodes->chunking();

        std::vector<std::size_t> chunkPos(3), blockBegin(3), blockEnd(3);
        for(const std::size_t chunkId : chunkIds) {

            // find the bounding box of this chunk and merge
            // with the full roi
            blocking.blockIdToBlockCoordinate(chunkId, chunkPos);
            blocking.getBlockBeginAndEnd(chunkPos, blockBegin, blockEnd);
            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis],
                                          static_cast<std::size_t>(blockBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis],
                                        static_cast<std::size_t>(blockEnd[axis]));
            }

            // load nodes from this chunk and insert into the node set
            if(!dsNodes->chunkExists(chunkPos)) {
                continue;
            }
            std::size_t nNodes;
            dsNodes->checkVarlenChunk(chunkPos, nNodes);
            std::vector<uint64_t> nodeSer(nNodes);
            dsNodes->readChunk(chunkPos, &nodeSer[0]);
            nodes.insert(nodeSer.begin(), nodeSer.end());

            // load edges from this chunk and insert into the edge set
            if(!dsEdges->chunkExists(chunkPos)) {
                continue;
            }
            std::size_t nEdges;
            dsEdges->checkVarlenChunk(chunkPos, nEdges);
            std::vector<uint64_t> edgeSer(nEdges);
            dsEdges->readChunk(chunkPos, &edgeSer[0]);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                edges.insert(std::make_pair(edgeSer[2 * edgeId], edgeSer[2 * edgeId + 1]));
            }

        }

    }


    inline void mergeSubgraphsMultiThreaded(const std::string & graphPath,
                                            const std::string & subgraphKey,
                                            const std::vector<std::size_t> & chunkIds,
                                            NodeSet & nodes,
                                            EdgeSet & edges,
                                            std::vector<std::size_t> & roiBegin,
                                            std::vector<std::size_t> & roiEnd,
                                            const int numberOfThreads) {
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        auto nThreads = threadpool.nThreads();

        // open the graph file
        z5::filesystem::handle::File graph(graphPath);

        // open the node and edge varlen dataset
        const std::string nodeKey = subgraphKey + "/nodes";
        const auto dsNodes = z5::openDataset(graph, nodeKey);
        const std::string edgeKey = subgraphKey + "/edges";
        const auto dsEdges = z5::openDataset(graph, edgeKey);

        // get blocking from dataset
        const auto & blocking = dsNodes->chunking();

        // initialize thread data
        struct PerThreadData {
            std::vector<std::size_t> roiBegin;
            std::vector<std::size_t> roiEnd;
            NodeSet nodes;
            EdgeSet edges;
        };
        std::vector<PerThreadData> threadData(nThreads);
        std::size_t maxSizeT = std::numeric_limits<std::size_t>::max();
        for(int t = 0; t < nThreads; ++t) {
            threadData[t].roiBegin = std::vector<std::size_t>({maxSizeT, maxSizeT, maxSizeT});
            threadData[t].roiEnd = std::vector<std::size_t>({0, 0, 0});
        }

        // merge nodes and edges multi threaded
        std::size_t nChunks = chunkIds.size();
        nifty::parallel::parallel_foreach(threadpool, nChunks, [&](const int tid,
                                                                   const int chunkIndex){

            // get the thread data
            const auto chunkId = chunkIds[chunkIndex];
            // for thread 0, we use the input sets instead of our thread data
            // to avoid one sequential merge in the end
            auto & threadNodes = (tid == 0) ? nodes : threadData[tid].nodes;
            auto & threadEdges = (tid == 0) ? edges : threadData[tid].edges;
            auto & threadBegin = threadData[tid].roiBegin;
            auto & threadEnd = threadData[tid].roiEnd;

            // find the bounding box of this chunk and merge
            // with the full roi
            std::vector<std::size_t> chunkPos(3), blockBegin(3), blockEnd(3);
            blocking.blockIdToBlockCoordinate(chunkId, chunkPos);
            blocking.getBlockBeginAndEnd(chunkPos, blockBegin, blockEnd);
            for(int axis = 0; axis < 3; ++axis) {
                roiBegin[axis] = std::min(roiBegin[axis],
                                          static_cast<std::size_t>(blockBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis],
                                        static_cast<std::size_t>(blockEnd[axis]));
            }

            // load nodes from this chunk and insert into the node set
            if(!dsNodes->chunkExists(chunkPos)) {
                return;
            }
            std::size_t nNodes;
            dsNodes->checkVarlenChunk(chunkPos, nNodes);
            std::vector<uint64_t> nodeSer(nNodes);
            dsNodes->readChunk(chunkPos, &nodeSer[0]);
            nodes.insert(nodeSer.begin(), nodeSer.end());

            // load edges from this chunk and insert into the edge set
            if(!dsEdges->chunkExists(chunkPos)) {
                return;
            }
            std::size_t nEdges;
            dsEdges->checkVarlenChunk(chunkPos, nEdges);
            std::vector<uint64_t> edgeSer(nEdges);
            dsEdges->readChunk(chunkPos, &edgeSer[0]);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                edges.insert(std::make_pair(edgeSer[2 * edgeId], edgeSer[2 * edgeId + 1]));
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
                roiBegin[axis] = std::min(roiBegin[axis],
                                          static_cast<std::size_t>(threadBegin[axis]));
                roiEnd[axis] = std::max(roiEnd[axis],
                                        static_cast<std::size_t>(threadEnd[axis]));
            }
        }
    }


    inline void mergeSubgraphs(const std::string & graphPath,
                               const std::string & subgraphKey,
                               const std::vector<std::size_t> & chunkIds,
                               const std::string & outKey,
                               const int numberOfThreads=1,
                               const bool serializeToVarlen=false) {
        // try unordered sets again here ?
        NodeSet nodes;
        EdgeSet edges;

        std::size_t maxSizeT = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> roiBegin({maxSizeT, maxSizeT, maxSizeT});
        std::vector<std::size_t> roiEnd({0, 0, 0});

        if(numberOfThreads == 1) {
            mergeSubgraphsSingleThreaded(graphPath, subgraphKey,
                                         chunkIds, nodes, edges,
                                         roiBegin, roiEnd);
        } else {
            mergeSubgraphsMultiThreaded(graphPath, subgraphKey,
                                        chunkIds, nodes, edges,
                                        roiBegin, roiEnd,
                                        numberOfThreads);
        }

        // serialize the merged graph
        if(serializeToVarlen) {
            serializeGraphToVarlen(graphPath, outKey,
                                   nodes, edges,
                                   roiBegin);
        } else {
            // we can only use compression for
            // big enough blocks (too small chunks will result in zlib error)
            // as a proxy we use the number of threads to determine if we use compression
            std::string compression = (numberOfThreads > 1) ? "gzip" : "raw";
            serializeGraph(graphPath, outKey,
                           nodes, edges,
                           roiBegin, roiEnd,
                           numberOfThreads,
                           compression);
        }
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                           const std::string & graphKey,
                           const std::string & subgraphKey,
                           const std::vector<std::size_t> & chunkIds,
                           const int numberOfThreads=1) {

        // we load the edges into a vector, because
        // it will be sorted by construction and we can take
        // advantage of O(logN) search with std::lower_bound
        std::vector<EdgeType> edges;
        z5::filesystem::handle::File graphFile(pathToGraph);
        loadEdges(pathToGraph, graphKey, edges, 0, numberOfThreads);

        // iterate over the blocks and insert the nodes and edges
        // construct threadpool
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        const auto nThreads = threadpool.nThreads();
        const std::size_t nChunks = chunkIds.size();

        // load the ege dataset and the edge id dataset
        const std::string edgeKey = subgraphKey + "/edges";
        const auto dsEdges = z5::openDataset(graphFile, edgeKey);
        const std::string edgeIdKey = subgraphKey + "/edge_ids";
        const auto dsEdgeIds = z5::openDataset(graphFile, edgeIdKey);

        // get blocking of dataset
        const auto & blocking = dsEdges->chunking();

        // handle all the chunks in parallel
        nifty::parallel::parallel_foreach(threadpool, nChunks, [&](const int tid,
                                                                   const int chunkIndex){
            // determine the chunk grid position
            const auto chunkId = chunkIds[chunkIndex];
            std::vector<std::size_t> chunkPos(3);
            blocking.blockIdToBlockCoordinate(chunkId, chunkPos);

            // load the edges from this chunk
            if(!dsEdges->chunkExists(chunkPos)) {
                return;
            }

            std::size_t nEdges;
            dsEdges->checkVarlenChunk(chunkPos, nEdges);
            std::vector<uint64_t> edgeSer(nEdges);
            dsEdges->readChunk(chunkPos, &edgeSer[0]);

            std::vector<EdgeType> blockEdges(nEdges / 2);
            for(std::size_t edgeId = 0; edgeId < nEdges / 2; ++edgeId) {
                blockEdges[edgeId] = std::make_pair(edgeSer[2 * edgeId],
                                                    edgeSer[2 * edgeId + 1]);
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
            dsEdgeIds->writeChunk(chunkPos, &edgeIds[0], true, edgeIds.size());
        });
    }


    inline void mapEdgeIds(const std::string & pathToGraph,
                    const std::string & graphKey,
                    const std::string & subgraphKey,
                    const std::size_t numberOfBlocks,
                    const int numberOfThreads=1) {
        std::vector<std::size_t> chunkIds(numberOfBlocks);
        std::iota(chunkIds.begin(), chunkIds.end(), 0);
        mapEdgeIds(pathToGraph, graphKey, subgraphKey, chunkIds, numberOfThreads);
    }

}
}
