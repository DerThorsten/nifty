#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/distributed/distributed_graph.hxx"
#include "nifty/tools/blocking.hxx"

// FIXME we should change hdf5 to xtensor backend
#include "nifty/hdf5/hdf5.hxx"
#include "nifty/hdf5/hdf5_array.hxx"

namespace nifty {
namespace distributed {


    inline void loadNiftyGraph(const std::string & graphPath,
                               nifty::graph::UndirectedGraph<> & g,
                               std::unordered_map<NodeType, NodeType> & relabeling,
                               const bool relabelNodes=true) {
        std::vector<NodeType> nodes;
        loadNodes(graphPath, nodes, 0);
        const size_t nNodes = nodes.size();

        std::vector<EdgeType> edges;
        loadEdges(graphPath, edges, 0);
        const size_t nEdges = edges.size();

        if(relabelNodes) {
            g.assign(nNodes, nEdges);
            for(size_t ii = 0; ii < nNodes; ++ii) {
                relabeling[nodes[ii]] = ii;
            }

            for(const auto & edge : edges) {
                g.insertEdge(relabeling[edge.first], relabeling[edge.second]);
            }

        } else {
            NodeType maxNode = *std::max_element(nodes.begin(), nodes.end());
            g.assign(maxNode + 1, nEdges);

            for(const auto & edge : edges) {
                g.insertEdge(edge.first, edge.second);
            }
        }

    }


    inline void nodeLabelingToPixels(const std::string & labelsPath,
                                     const std::string & outPath,
                                     const xt::xtensor<NodeType, 1> & nodeLabeling,
                                     const std::vector<size_t> & blockIds,
                                     const std::vector<size_t> & blockShape) {
        // in and out dataset
        auto labelDs = z5::openDataset(labelsPath);
        auto outDs = z5::openDataset(outPath);
        Shape3Type arrayShape = {blockShape[0], blockShape[1], blockShape[2]};
        xt::xtensor<NodeType, 3> labels(arrayShape);

        // blocking
        CoordType roiBegin = {0, 0, 0};
        CoordType shape = {labelDs->shape(0), labelDs->shape(1), labelDs->shape(2)};
        CoordType bShape = {blockShape[0], blockShape[1], blockShape[2]};
        nifty::tools::Blocking<3> blocking(roiBegin, shape, bShape);

        for(auto blockId : blockIds) {
            // get block and roi
            const auto & block = blocking.getBlock(blockId);
            const auto & begin = block.begin();
            const auto & end = block.end();

            // actual block shape and resizeing
            std::vector<size_t> actualBlockShape(3);
            bool needsResize = false;
            for(unsigned axis = 0; axis < 3; ++axis) {
                actualBlockShape[axis] = end[axis] - begin[axis];
                if(actualBlockShape[axis] != labels.shape()[axis]) {
                    needsResize = true;
                }
            }
            if(needsResize) {
                labels.reshape(actualBlockShape);
            }

            // get labels and do the mapping
            z5::multiarray::readSubarray<NodeType>(labelDs, labels, begin.begin());
            for(size_t z = 0; z < actualBlockShape[0]; ++z) {
                for(size_t y = 0; y < actualBlockShape[1]; ++y) {
                    for(size_t x = 0; x < actualBlockShape[2]; ++x) {
                        labels(z, y, x) = nodeLabeling(labels(z, y, x));
                    }
                }
            }

            // write out
            z5::multiarray::writeSubarray<NodeType>(outDs, labels, begin.begin());
        }
    }


    inline void nodesToBlocks(const std::string & graphBlockPrefix,
                              const std::string & outNodePath,
                              const size_t numberOfBlocks,
                              const size_t numberOfNodes,
                              const int numberOfThreads) {

        typedef std::vector<size_t> BlockStorage;
        typedef std::vector<BlockStorage> NodeBlockStorage;

        // make per thread data
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        const size_t nThreads = threadpool.nThreads();
        std::vector<NodeBlockStorage> perThreadData(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&perThreadData, numberOfNodes](const int tId, const int threadId) {
            perThreadData[threadId] = NodeBlockStorage(numberOfNodes);
        });

        // map nodes to blocks multithreaded
        nifty::parallel::parallel_foreach(threadpool, numberOfBlocks, [&](const int tId, const size_t blockId){
            std::vector<NodeType> blockNodes;
            const std::string blockPath = graphBlockPrefix + std::to_string(blockId);
            loadNodes(blockPath, blockNodes, 0);
            auto & nodeVector = perThreadData[tId];
            for(NodeType node : blockNodes) {
                nodeVector[node].push_back(blockId);
            }
        });

        // merge the results
        auto & nodeVector = perThreadData[0];
        nifty::parallel::parallel_foreach(threadpool, numberOfNodes, [&](const int tId, const NodeType node){
            auto & blockVector = nodeVector[node];
            for(int threadId = 1; threadId < nThreads; ++threadId) {
                const auto & threadVector = perThreadData[threadId][node];
                blockVector.insert(blockVector.end(), threadVector.begin(), threadVector.end());
            }
            std::sort(blockVector.begin(), blockVector.end());
        });

        // we use hdf5 here, because n5 would result in too many files (1 file per node)
        const auto h5File = nifty::hdf5::openFile(outNodePath);
        const std::vector<size_t> zero1Coord({0});
        nifty::parallel::parallel_foreach(threadpool, numberOfNodes, [&](const int tId, const NodeType node){
            const std::string nodeKey = "node_" + std::to_string(node);
            const auto & blockVector = nodeVector[node];
            const std::vector<size_t> shape = {blockVector.size()};
            auto nodeDs = nifty::hdf5::Hdf5Array<NodeType>(h5File, nodeKey,
                                                           shape.begin(), shape.end(),
                                                           shape.begin());
            // FIXME change hdf5 to xtensor backend
            marray::Marray<NodeType> blockArray(shape.begin(), shape.end());
            for(size_t ii = 0; ii < shape[0]; ++ii) {
                blockArray(ii) = blockVector[ii];
            }
            nodeDs.writeSubarray(zero1Coord.begin(), blockArray);
        });
    }


    template<class NODE_ARRAY>
    inline void nodesToBlocksWithLabeling(const size_t numberOfNewNodes,
                                          const nifty::tools::Blocking<3> & blocking,
                                          const nifty::tools::Blocking<3> & newBlocking,
                                          const std::string & graphBlockPrefix,
                                          const xt::xexpression<NODE_ARRAY> & nodeLabelingExp,
                                          const std::string & nodeOutPath,
                                          std::vector<std::set<NodeType>> & blockNodeStorage,
                                          nifty::parallel::ThreadPool & threadpool) {

        typedef std::set<size_t> BlockStorage;
        typedef std::vector<BlockStorage> NodeBlockStorage;
        const auto & nodeLabeling = nodeLabelingExp.derived_cast();

        // make per thread data
        const size_t nThreads = threadpool.nThreads();
        std::vector<NodeBlockStorage> perThreadData(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&perThreadData, numberOfNewNodes](const int tId, const int threadId) {
            perThreadData[threadId] = NodeBlockStorage(numberOfNewNodes);
        });

        // map nodes to blocks multithreaded
        const size_t numberOfNewBlocks = newBlocking.numberOfBlocks();
        nifty::parallel::parallel_foreach(threadpool, numberOfNewBlocks, [&](const int tId, const size_t blockId){
            // out data
            auto & newBlockNodes = blockNodeStorage[blockId];
            auto & nodeVector = perThreadData[tId];

            // find the relevant old blocks
            const auto & newBlock = newBlocking.getBlock(blockId);
            std::vector<size_t> oldBlockIds;
            blocking.getBlockIdsInBoundingBox(newBlock.begin(), newBlock.end(),
                                              {0L, 0L, 0L}, oldBlockIds);

            // iterate over the old blocks and write out all the nodes
            for(auto oldBlockId : oldBlockIds) {
                const std::string blockPath = graphBlockPrefix + std::to_string(oldBlockId);
                std::vector<NodeType> blockNodes;
                loadNodes(blockPath, blockNodes, 0);
                for(const NodeType node : blockNodes) {
                    const NodeType newNode = nodeLabeling(node);
                    nodeVector[newNode].insert(blockId);
                    newBlockNodes.insert(newNode);
                }
            }
        });

        // merge the results
        auto & nodeVector = perThreadData[0];
        nifty::parallel::parallel_foreach(threadpool, numberOfNewNodes, [&](const int tId, const NodeType node){
            auto & blockStorage = nodeVector[node];
            for(int threadId = 1; threadId < nThreads; ++threadId) {
                const auto & threadStorage = perThreadData[threadId][node];
                blockStorage.insert(threadStorage.begin(), threadStorage.end());
            }
        });

        // we use hdf5 here, because n5 would result in too many files (1 file per node)
        const auto h5File = nifty::hdf5::openFile(nodeOutPath);
        const std::vector<size_t> zero1Coord({0});
        nifty::parallel::parallel_foreach(threadpool, numberOfNewNodes, [&](const int tId, const NodeType node){

            const std::string nodeKey = "node_" + std::to_string(node);
            const auto & blockStorage = nodeVector[node];
            std::vector<size_t> blockVector(blockStorage.begin(), blockStorage.end());

            const std::vector<size_t> shape = {blockVector.size()};
            auto nodeDs = nifty::hdf5::Hdf5Array<NodeType>(h5File, nodeKey,
                                                           shape.begin(), shape.end(),
                                                           shape.begin());
            // FIXME change hdf5 to xtensor backend
            marray::Marray<NodeType> blockArray(shape.begin(), shape.end());
            for(size_t ii = 0; ii < shape[0]; ++ii) {
                blockArray(ii) = blockVector[ii];
            }
            nodeDs.writeSubarray(zero1Coord.begin(), blockArray);
        });
    }


    template<class NODE_ARRAY, class EDGE_ARRAY>
    inline void serializeMergedGraph(const std::string & graphBlockPrefix,
                                     const CoordType & shape,
                                     const CoordType & blockShape,
                                     const CoordType & newBlockShape,
                                     const size_t numberOfNewNodes,
                                     const xt::xexpression<NODE_ARRAY> & nodeLabelingExp,
                                     const xt::xexpression<EDGE_ARRAY> & edgeLabelingExp,
                                     const std::string & nodeOutPrefix,
                                     const std::string & graphOutPrefix,
                                     const int numberOfThreads) {

        const auto & nodeLabeling = nodeLabelingExp.derived_cast();
        const auto & edgeLabeling = edgeLabelingExp.derived_cast();
        nifty::parallel::ThreadPool threadpool(numberOfThreads);

        const CoordType roiBegin = {0, 0, 0};
        nifty::tools::Blocking<3> blocking(roiBegin, shape, blockShape);
        nifty::tools::Blocking<3> newBlocking(roiBegin, shape, newBlockShape);

        const size_t numberOfNewBlocks = newBlocking.numberOfBlocks();
        std::vector<std::set<NodeType>> blockNodeStorage(numberOfNewBlocks);
        // load new nodes and serialize the new node to block assignment
        nodesToBlocksWithLabeling(numberOfNewNodes,
                                  blocking,
                                  newBlocking,
                                  graphBlockPrefix,
                                  nodeLabelingExp,
                                  nodeOutPrefix,
                                  blockNodeStorage,
                                  threadpool);

        // serialize the merged sub-graphs
        const std::vector<size_t> zero1Coord({0});
        const std::vector<size_t> zero2Coord({0, 0});
        nifty::parallel::parallel_foreach(threadpool, numberOfNewBlocks, [&](const int tId, const size_t blockId){
            // create the out group
            const std::string outPath = graphOutPrefix + std::to_string(blockId);
            z5::handle::Group group(outPath);
            z5::createGroup(group, false);

            // get the new block node ids and serialize them
            const auto & blockNodes = blockNodeStorage[blockId];
            std::vector<size_t> nodeShape = {blockNodes.size()};
            auto dsNodes = z5::createDataset(group, "nodes", "uint64", nodeShape, nodeShape, false);
            Shape1Type nodeSerShape = {blockNodes.size()};
            Tensor1 nodeSer(nodeSerShape);
            size_t i = 0;
            for(const auto node : blockNodes) {
                nodeSer(i) = node;
                ++i;
            }
            z5::multiarray::writeSubarray<NodeType>(dsNodes, nodeSer, zero1Coord.begin());

            // find the relevant old blocks
            const auto & newBlock = newBlocking.getBlock(blockId);
            std::vector<size_t> oldBlockIds;
            blocking.getBlockIdsInBoundingBox(newBlock.begin(), newBlock.end(),
                                              {0L, 0L, 0L}, oldBlockIds);

            // iterate over the old blocks and load all edges and edge ods
            std::map<EdgeIndexType, EdgeType> newEdges;
            for(auto oldBlockId : oldBlockIds) {
                const std::string blockPath = graphBlockPrefix + std::to_string(oldBlockId);
                std::vector<EdgeType> subEdges;
                std::vector<EdgeIndexType> subEdgeIds;
                loadEdges(blockPath, subEdges, 0);
                loadEdgeIndices(blockPath, subEdgeIds, 0);

                // map edges and edge ids to the merged graph and serialize
                for(size_t ii = 0; ii < subEdges.size(); ++ii) {
                    const auto newEdgeId = edgeLabeling(subEdgeIds[ii]);
                    if(newEdgeId != -1) {
                        const EdgeType & uv = subEdges[ii];
                        const NodeType newU = nodeLabeling(uv.first);
                        const NodeType newV = nodeLabeling(uv.second);
                        newEdges[newEdgeId] = std::make_pair(newU, newV);
                    }
                }
            }

            const size_t nNewNodes = blockNodes.size();
            const size_t nNewEdges = newEdges.size();
            // serialize the new edges and the new edge ids
            if(nNewEdges > 0) {

                Shape2Type edgeSerShape = {nNewEdges, 2};
                Tensor2 edgeSer(edgeSerShape);

                Shape1Type edgeIdSerShape = {nNewEdges};
                Tensor1 edgeIdSer(edgeIdSerShape);

                size_t i = 0;
                for(const auto & edge : newEdges) {
                    edgeIdSer(i) = edge.first;
                    edgeSer(i, 0) = edge.second.first;
                    edgeSer(i, 1) = edge.second.second;
                    ++i;
                }

                // serialize the edges
                std::vector<size_t> edgeShape = {nNewEdges, 2};
                auto dsEdges = z5::createDataset(group, "edges", "uint64", edgeShape, edgeShape, false);
                z5::multiarray::writeSubarray<NodeType>(dsEdges, edgeSer, zero2Coord.begin());

                // serialize the edge ids
                std::vector<size_t> edgeIdShape = {nNewEdges};
                auto dsEdgeIds = z5::createDataset(group, "edgeIds", "int64", edgeIdShape, edgeIdShape, false);
                z5::multiarray::writeSubarray<EdgeIndexType>(dsEdgeIds, edgeIdSer, zero1Coord.begin());
            }

            // serialize metadata (number of edges and nodes and position of the block)
            nlohmann::json attrs;
            attrs["numberOfNodes"] = nNewNodes;
            attrs["numberOfEdges"] = nNewEdges;
            // TODO ideally we would get the rois from the prev. graph block too, but I am too lazy right now
            // attrs["roiBegin"] = std::vector<size_t>(roiBegin.begin(), roiBegin.end());
            // attrs["roiEnd"] = std::vector<size_t>(roiEnd.begin(), roiEnd.end());

            z5::writeAttributes(group, attrs);
        });
    }


    template<class NODE_ARRAY>
    inline void extractSubgraphFromNodes(const xt::xexpression<NODE_ARRAY> & nodesExp,
                                         const std::string & nodeStorage,
                                         const std::string & graphBlockPrefix,
                                         std::vector<EdgeType> & uvIdsOut,
                                         std::vector<EdgeIndexType> & innerEdgesOut,
                                         std::vector<EdgeIndexType> & outerEdgesOut) {
        //
        const auto & nodes = nodesExp.derived_cast();

        // find all blocks that have overlap with the nodes
        std::set<size_t> blocks;
        const auto h5File = nifty::hdf5::openFile(nodeStorage);
        const std::vector<size_t> zero1Coord({0});

        for(const NodeType node : nodes) {
            const std::string nodeKey = "node_" + std::to_string(node);
            auto nodeDs = nifty::hdf5::Hdf5Array<NodeType>(h5File, nodeKey);
            nifty::marray::Marray<NodeType> nodeBlockIds(nodeDs.shape().begin(), nodeDs.shape().end());

            nodeDs.readSubarray(zero1Coord.begin(), nodeBlockIds);
            blocks.insert(nodeBlockIds.begin(), nodeBlockIds.end());
        }

        // extract the (distributed) graph and edge ids
        std::vector<std::string> blockList;
        for(auto block : blocks) {
            blockList.emplace_back(graphBlockPrefix + std::to_string(block));
        }
        std::vector<EdgeIndexType> edgeIds;
        const Graph g(blockList, edgeIds);

        // extract the subgraph uv-ids (with dense node labels)
        // as well as inner and outer edges associated with the node list

        // first find the mapping to dense node index
        std::unordered_map<NodeType, NodeType> nodeMapping;
        for(size_t i = 0; i < nodes.size(); ++i) {
            nodeMapping[nodes(i)] = i;
        }

        // then iterate over the adjacency and extract inner and outer edges
        for(const NodeType u : nodes) {
            const auto & uAdjacency = g.nodeAdjacency(u);
            for(const auto & adj : uAdjacency) {
                const NodeType v = adj.first;
                const EdgeIndexType edge = adj.second;
                // we do the look-up in the node-mapping instead of the node-list, because it's a hash-map
                // (and thus faster than array lookup)
                if(nodeMapping.find(v) != nodeMapping.end()) {
                    // we will encounter inner edges twice, so we only add them for u < v
                    if(u < v) {
                        innerEdgesOut.push_back(edgeIds[edge]);
                        uvIdsOut.emplace_back(std::make_pair(nodeMapping[u], nodeMapping[v]));
                    }
                } else {
                    // outer edges occur only once by construction
                    outerEdgesOut.push_back(edgeIds[edge]);
                }
            }
        }
    }

}
}
