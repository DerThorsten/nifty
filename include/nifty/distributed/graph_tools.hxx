#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/distributed/distributed_graph.hxx"
#include "nifty/tools/blocking.hxx"

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


    // TODO change node storage to hdf5
    inline void nodesToBlocks(const std::string & graphBlockPrefix,
                              const std::string & outNodePrefix,
                              const size_t numberOfBlocks,
                              const size_t numberOfNodes,
                              const int numberOfThreads) {

        typedef std::vector<size_t> BlockStorage;
        typedef std::vector<BlockStorage> NodeBlockStorage;

        // make per thread data
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        const size_t nThreads = threadpool.nThreads();
        std::vector<NodeBlockStorage> perThreadData(nThreads);
        nifty::parallel::parallel_foreach(threadpool, nThreads, [&perThreadData, numberOfBlocks](const int tId, const int threadId) {
            perThreadData[threadId] = NodeBlockStorage(numberOfBlocks);
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

        // TODO maybe we should write this as hdf5, because for a
        // large number of nodes, this will create a lot of files
        // write each node vector to its own dataset
        const std::vector<size_t> chunk = {0};
        nifty::parallel::parallel_foreach(threadpool, numberOfNodes, [&](const int tId, const NodeType node){
            const std::string nodePath = outNodePrefix + std::to_string(node);
            const auto & blockVector = nodeVector[node];
            const std::vector<size_t> shape = {blockVector.size()};
            auto nodeDs = z5::createDataset(nodePath, "uint64", shape, shape, false);
            nodeDs->writeChunk(chunk, &blockVector[0]);
        });
    }


    // TODO change node storage to hdf5
    template<class NODE_ARRAY>
    inline void extractSubgraphFromNodes(const xt::xexpression<NODE_ARRAY> & nodesExp,
                                         const std::string & nodeStoragePrefix,
                                         const std::string & graphBlockPrefix,
                                         nifty::graph::UndirectedGraph<EdgeIndexType, NodeType> & graphOut,
                                         std::vector<EdgeIndexType> & innerEdgesOut,
                                         std::vector<EdgeIndexType> & outerEdgesOut) {
        //
        const auto & nodes = nodesExp.derived_cast();

        // find all blocks that have overlap with the nodes
        std::vector<size_t> chunk = {0};
        std::set<size_t> blocks;
        for(const NodeType node : nodes) {
            const std::string nodePath = nodeStoragePrefix + std::to_string(node);
            // TODO change node storage to hdf5
            auto nodeDs = z5::openDataset(nodePath);
            const size_t nBlocks = nodeDs->maxChunkShape()[0]; // we only have a single chunk, so this is fine
            std::vector<size_t> nodeBlockIds(nBlocks);
            nodeDs->readChunk(chunk, &nodeBlockIds[0]);
            blocks.insert(nodeBlockIds.begin(), nodeBlockIds.end());
        }

        // extract the (distributed) graph
        std::vector<std::string> blockList;
        for(auto block : blocks) {
            blockList.emplace_back(graphBlockPrefix + std::to_string(block));
        }
        const Graph g(blockList);

        // extract the global edge ids associated with this graph from the blocks
        std::vector<EdgeIndexType> edgeIds;
        for(auto block : blocks) {
            const std::string graphPath = graphBlockPrefix + std::to_string(block);
            loadEdgeIndices(graphPath, edgeIds, edgeIds.size());
        }
        // make the edge ids unique
        std::sort(edgeIds.begin(), edgeIds.end());
        edgeIds.resize(std::unique(edgeIds.begin(), edgeIds.end()) - edgeIds.begin());

        // extract the subgraph (as nifty undirected graph)
        // as well as inner and outer edges associated with the node list

        // first find the mapping to dense node index
        std::unordered_map<NodeType, NodeType> nodeMapping;
        for(size_t i = 0; i < nodes.size(); ++i) {
            nodeMapping[nodes(i)] = i;
        }

        // then iterate over the adjacency and extract inner and outer edges
        std::vector<EdgeType> edges;
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
                        innerEdgesOut.push_back(edge);
                        edges.emplace_back(std::make_pair(u, v));
                    }
                    else {
                        // outer edges occur only once by construction
                        outerEdgesOut.push_back(edge);
                    }
                }
            }
        }

        // construct the output graph
        graphOut.assign(nodes.shape()[0], innerEdgesOut.size());
        for(std::size_t i = 0; i < innerEdgesOut.size(); ++i) {
            const auto & uv = edges[i];
            graphOut.insertEdge(nodeMapping[uv.first], nodeMapping[uv.second]);
        }
    }

}
}
