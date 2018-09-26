#pragma once

#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/distributed/distributed_graph.hxx"
#include "nifty/tools/blocking.hxx"

namespace fs = boost::filesystem;

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
                labels.resize(actualBlockShape);
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


    template<class NODE_ARRAY, class EDGE_ARRAY>
    inline void serializeMergedGraph(const std::string & graphBlockPrefix,
                                     const CoordType & shape,
                                     const CoordType & blockShape,
                                     const CoordType & newBlockShape,
                                     const std::vector<size_t> & newBlockIds,
                                     const size_t numberOfNewNodes,
                                     const xt::xexpression<NODE_ARRAY> & nodeLabelingExp,
                                     const xt::xexpression<EDGE_ARRAY> & edgeLabelingExp,
                                     const std::string & graphOutPrefix,
                                     const int numberOfThreads) {

        const auto & nodeLabeling = nodeLabelingExp.derived_cast();
        const auto & edgeLabeling = edgeLabelingExp.derived_cast();
        nifty::parallel::ThreadPool threadpool(numberOfThreads);

        const CoordType roiBegin = {0, 0, 0};
        nifty::tools::Blocking<3> blocking(roiBegin, shape, blockShape);
        nifty::tools::Blocking<3> newBlocking(roiBegin, shape, newBlockShape);

        // const size_t numberOfNewBlocks = newBlocking.numberOfBlocks();
        const size_t numberOfNewBlocks = newBlockIds.size();
        std::vector<std::set<NodeType>> blockNodeStorage(numberOfNewBlocks);

        // load new nodes
        nifty::parallel::parallel_foreach(threadpool,
                                          numberOfNewBlocks, [&](const int tId,
                                                                 const size_t blockIndex){
            const size_t blockId = newBlockIds[blockIndex];
            auto & newBlockNodes = blockNodeStorage[blockId];

            // find the relevant old blocks
            const auto & newBlock = newBlocking.getBlock(blockId);
            std::vector<size_t> oldBlockIds;
            blocking.getBlockIdsInBoundingBox(newBlock.begin(), newBlock.end(), oldBlockIds);

            // iterate over the old blocks and write out all the nodes
            for(auto oldBlockId : oldBlockIds) {
                const std::string blockPath = graphBlockPrefix + std::to_string(oldBlockId);


                // if we are dealing with region of interests, the sub-graph might actually not exist
                // so we need to check and skip if it does not exist.
                if(!fs::exists(blockPath)) {
                    continue;
                }

                std::vector<NodeType> blockNodes;
                loadNodes(blockPath, blockNodes, 0);
                for(const NodeType node : blockNodes) {
                    const NodeType newNode = nodeLabeling(node);
                    newBlockNodes.insert(newNode);
                }
            }
        });

        // serialize the merged sub-graphs
        const std::vector<size_t> zero1Coord({0});
        const std::vector<size_t> zero2Coord({0, 0});
        nifty::parallel::parallel_foreach(threadpool,
                                          numberOfNewBlocks, [&](const int tId,
                                                                 const size_t blockId){
            // create the out group
            const std::string outPath = graphOutPrefix + std::to_string(blockId);
            z5::handle::Group group(outPath);
            z5::createGroup(group, false);

            // get the new block node ids and serialize them
            const auto & blockNodes = blockNodeStorage[blockId];
            std::vector<size_t> nodeShape = {blockNodes.size()};
            auto dsNodes = z5::createDataset(group, "nodes", "uint64",
                                             nodeShape, nodeShape, false);
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
            blocking.getBlockIdsInBoundingBox(newBlock.begin(), newBlock.end(), oldBlockIds);

            // iterate over the old blocks and load all edges and edge ods
            std::map<EdgeIndexType, EdgeType> newEdges;
            for(auto oldBlockId : oldBlockIds) {
                const std::string blockPath = graphBlockPrefix + std::to_string(oldBlockId);

                // if we are dealing with region of interests, the sub-graph might actually not exist
                // so we need to check and skip if it does not exist.
                if(!fs::exists(blockPath)) {
                    continue;
                }

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
                auto dsEdges = z5::createDataset(group, "edges", "uint64",
                                                 edgeShape, edgeShape, false);
                z5::multiarray::writeSubarray<NodeType>(dsEdges, edgeSer, zero2Coord.begin());

                // serialize the edge ids
                std::vector<size_t> edgeIdShape = {nNewEdges};
                auto dsEdgeIds = z5::createDataset(group, "edgeIds", "int64",
                                                   edgeIdShape, edgeIdShape, false);
                z5::multiarray::writeSubarray<EdgeIndexType>(dsEdgeIds, edgeIdSer,
                                                             zero1Coord.begin());
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


    // we have to look at surprisingly many blocks, which makes
    // this function pretty inefficient to
    // FIXME I am not 100 % sure if this is not due to some bug
    template<class NODE_ARRAY>
    inline void extractSubgraphFromNodes(const xt::xexpression<NODE_ARRAY> & nodesExp,
                                         const std::string & graphBlockPrefix,
                                         const CoordType & shape,
                                         const CoordType & blockShape,
                                         const size_t startBlockId,
                                         std::vector<EdgeType> & uvIdsOut,
                                         std::vector<EdgeIndexType> & innerEdgesOut,
                                         std::vector<EdgeIndexType> & outerEdgesOut) {
        //
        const auto & nodes = nodesExp.derived_cast();

        // TODO refactor this part
        nifty::tools::Blocking<3> blocking({0L, 0L, 0L}, shape, blockShape);

        // find all blocks that have overlap with the nodes
        // beginning from the start block id and adding all neighbors, until nodes are no
        // longer present
        std::vector<size_t> blockVector = {startBlockId};
        std::unordered_set<int64_t> blocksProcessed;
        blocksProcessed.insert(startBlockId);

        const std::vector<bool> dirs = {false, true};
        std::queue<int64_t> blockQueue;
        // first, we enqueue all the neighboring blocks to the start block
        for(unsigned axis = 0; axis < 3; ++axis) {
            for(const bool lower : dirs) {
                const int64_t neighborId = blocking.getNeighborId(startBlockId, axis, lower);
                if(neighborId != -1) {
                    blockQueue.push(neighborId);
                }
            }
        }

        while(!blockQueue.empty()) {
            const int64_t blockId = blockQueue.front();
            blockQueue.pop();

            // check if we have already looked at this block
            if(blocksProcessed.find(blockId) != blocksProcessed.end()) {
                continue;
            }

            std::vector<NodeType> blockNodes;
            const std::string blockPath = graphBlockPrefix + std::to_string(blockId);
            // load the nodes in this block
            loadNodes(blockPath, blockNodes, 0);
            bool haveNode = false;

            // iterate over the node list and check if any of them is in the block
            for(const NodeType node: nodes) {
                // the node lists are sorted, hence we can use binary search
                auto it = std::lower_bound(blockNodes.begin(), blockNodes.end(), node);
                if(it != blockNodes.end()) {
                    haveNode = true;
                    break;
                }
            }

            // mark this block as processed
            blocksProcessed.insert(blockId);
            // if we have one of the nodes, push back the block id and
            // enqueue the neighbors
            if(haveNode) {
                blockVector.push_back(blockId);
                for(unsigned axis = 0; axis < 3; ++axis) {
                    for(const bool lower : dirs) {
                        const int64_t neighborId = blocking.getNeighborId(blockId, axis, lower);
                        if(neighborId != -1) {
                            blockQueue.push(neighborId);
                        }
                    }
                }
            }
        }

        // extract the (distributed) graph and edge ids
        std::vector<std::string> blockList;
        for(auto block : blockVector) {
            blockList.emplace_back(graphBlockPrefix + std::to_string(block));
        }
        std::set<size_t> unBlocks(blockVector.begin(), blockVector.end());

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


    template<class LABELS, class VALUES, class OVERLAPS>
    inline void computeLabelOverlaps(const xt::xexpression<LABELS> & labelsExp,
                                     const xt::xexpression<VALUES> & valuesExp,
                                     OVERLAPS & overlaps) {
        const auto & labels = labelsExp.derived_cast();
        const auto & values = valuesExp.derived_cast();

        CoordType shape;
        std::copy(labels.shape().begin(), labels.shape().end(), shape.begin());

        nifty::tools::forEachCoordinate(shape, [&](const CoordType & coord){
            const auto node = xtensor::read(labels, coord);
            const auto l = xtensor::read(values, coord);
            auto ovlpIt = overlaps.find(node);
            if(ovlpIt == overlaps.end()) {
                overlaps.emplace(node, std::unordered_map<uint64_t, size_t>{{l, 1}});
            }
            else {
                // FIXME not sure how this can work
                ovlpIt->second[l] += 1;
            }
        });
    }

}
}
