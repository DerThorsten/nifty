#pragma once

#include "boost/pending/disjoint_sets.hpp"
#include "z5/util/for_each.hxx"
#include "z5/util/util.hxx"

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


    // FIXME this sometimes fails with a floating point exception, but not really reproducible
    template<class NODE_ARRAY, class EDGE_ARRAY>
    inline void serializeMergedGraph(const std::string & graphBlockPrefix,
                                     const CoordType & shape,
                                     const CoordType & blockShape,
                                     const CoordType & newBlockShape,
                                     const std::vector<size_t> & newBlockIds,
                                     const xt::xexpression<NODE_ARRAY> & nodeLabelingExp,
                                     const xt::xexpression<EDGE_ARRAY> & edgeLabelingExp,
                                     const std::string & graphOutPrefix,
                                     const int numberOfThreads,
                                     const bool serializeEdges) {

        typedef std::set<NodeType> BlockNodeStorage;

        const auto & nodeLabeling = nodeLabelingExp.derived_cast();
        const auto & edgeLabeling = edgeLabelingExp.derived_cast();
        nifty::parallel::ThreadPool threadpool(numberOfThreads);

        const CoordType roiBegin = {0, 0, 0};
        nifty::tools::Blocking<3> blocking(roiBegin, shape, blockShape);
        nifty::tools::Blocking<3> newBlocking(roiBegin, shape, newBlockShape);

        const size_t numberOfNewBlocks = newBlockIds.size();

        // serialize the merged sub-graphs
        const std::vector<size_t> zero1Coord({0});
        const std::vector<size_t> zero2Coord({0, 0});
        nifty::parallel::parallel_foreach(threadpool,
                                          numberOfNewBlocks, [&](const int tId,
                                                                 const size_t blockIndex){
            const size_t blockId = newBlockIds[blockIndex];
            BlockNodeStorage newBlockNodes;

            // find the relevant old blocks
            const auto & newBlock = newBlocking.getBlock(blockId);
            std::vector<size_t> oldBlockIds;
            blocking.getBlockIdsInBoundingBox(newBlock.begin(), newBlock.end(), oldBlockIds);

            // iterate over the old blocks and find all nodes
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
                    newBlockNodes.insert(nodeLabeling(node));
                }
            }

            // create the out group
            const std::string outPath = graphOutPrefix + std::to_string(blockId);
            z5::handle::Group group(outPath);
            z5::createGroup(group, false);

            // serialize the new nodes
            const size_t nNewNodes = newBlockNodes.size();
            std::vector<size_t> nodeShape = {nNewNodes};
            auto dsNodes = z5::createDataset(group, "nodes", "uint64",
                                             nodeShape, nodeShape, false);
            Shape1Type nodeSerShape = {nNewNodes};
            Tensor1 nodeSer(nodeSerShape);
            size_t i = 0;
            for(const auto node : newBlockNodes) {
                nodeSer(i) = node;
                ++i;
            }
            z5::multiarray::writeSubarray<NodeType>(dsNodes, nodeSer, zero1Coord.begin());

            if(!serializeEdges) {
                // serialize metadata (number of edges and nodes and position of the block)
                nlohmann::json attrs;
                attrs["numberOfNodes"] = nNewNodes;
                z5::writeAttributes(group, attrs);
                return;
            }

            // iterate over the old blocks and load all edges and edge ids
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
    // this function pretty inefficient
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
    inline void computeLabelOverlaps(const LABELS & labels,
                                     const VALUES & values,
                                     OVERLAPS & overlaps) {
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
                ovlpIt->second[l] += 1;
            }
        });
    }


    template<class LABELS, class VALUES>
    inline void serializeLabelOverlaps(const LABELS & labels,
                                       const VALUES & values,
                                       const std::string & dsPath,
                                       const std::vector<std::size_t> & chunkId) {
        typedef typename LABELS::value_type LabelType;
        typedef typename VALUES::value_type ValueType;
        typedef std::unordered_map<ValueType, std::size_t> OverlapType;
        // extract the overlaps
        std::unordered_map<LabelType, OverlapType> overlaps;
        computeLabelOverlaps(labels, values, overlaps);

        // serialize the overlaps

        // first determine the serialization size
        std::size_t serSize = 0;
        for(const auto & elem: overlaps) {
            // per label, we serialize labelId, number of values,
            // the values and value-counts
            serSize += 2 + 2 * elem.second.size();
        }

        // make serialize
        std::vector<uint64_t> serialization(serSize);
        std::size_t serPos = 0;
        for(const auto & elem: overlaps) {
            const uint64_t labelId = static_cast<uint64_t>(elem.first);
            serialization[serPos] = labelId;
            ++serPos;

            const uint64_t count = elem.second.size();
            serialization[serPos] = count;
            ++serPos;

            for(const auto & ovlp: elem.second) {
                const uint64_t value = static_cast<uint64_t>(ovlp.first);
                serialization[serPos] = value;
                ++serPos;

                const uint64_t count = static_cast<uint64_t>(ovlp.second);
                serialization[serPos] = count;
                ++serPos;
            }
        }

        // write serialization
        auto ds = z5::openDataset(dsPath);
        ds->writeChunk(chunkId, &serialization[0], true, serSize);
    }


    inline void mergeAndSerializeOverlaps(const std::string & inputPath,
                                          const std::string & outputPath,
                                          const bool max_overlap,
                                          const int numberOfThreads) {

        typedef std::unordered_map<uint64_t, std::size_t> OverlapType;
        typedef std::unordered_map<uint64_t, OverlapType> LabelToOverlaps;

        std::vector<LabelToOverlaps> threadData(numberOfThreads);
        std::vector<uint64_t> threadMax(numberOfThreads, 0);

        auto inputDs = z5::openDataset(inputPath);
        z5::util::parallel_for_each_chunk(*inputDs,
                                          numberOfThreads,
                                          [&threadData,
                                           &threadMax](const int tId,
                                                       const z5::Dataset & ds,
                                                       const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            z5::handle::Chunk chunk(ds.handle(), chunkCoord, ds.isZarr());
            if(!chunk.exists()) {
                return;
            }
            bool isVarlen;
            const std::size_t chunkSize = ds.getDiscChunkSize(chunkCoord, isVarlen);
            std::vector<uint64_t> chunkOverlaps(chunkSize);
            ds.readChunk(chunkCoord, &chunkOverlaps[0]);

            // deserialize the data
            auto & thisData = threadData[tId];
            auto & thisMax = threadMax[tId];

            std::size_t pos = 0;
            while(pos < chunkSize) {
                const uint64_t labelId = chunkOverlaps[pos];
                ++pos;

                if(labelId > thisMax) {
                    thisMax = labelId;
                }

                const uint64_t nValues = chunkOverlaps[pos];
                ++pos;

                auto labelIt = thisData.find(labelId);
                if(labelIt == thisData.end()) {
                    labelIt = thisData.emplace(std::make_pair(labelId, OverlapType())).first;
                }

                auto ovlps = labelIt->second;

                for(size_t i = 0; i < nValues; ++i) {
                    const uint64_t value = chunkOverlaps[pos];
                    ++pos;

                    auto valIt = ovlps.find(value);
                    if(valIt == ovlps.end()) {
                        valIt = ovlps.emplace(std::make_pair(value, 0)).first;
                    }

                    const uint64_t count = chunkOverlaps[pos];
                    ++pos;

                    valIt->second += count;
                }
            }
        });

        // find the upper label bound
        const uint64_t nLabels = *std::max_element(threadMax.begin(), threadMax.end()) + 1;

        // merge the thread data
        std::vector<OverlapType> overlaps(nLabels);
        nifty::parallel::parallel_foreach(numberOfThreads,
                                          nLabels,
                                          [&threadData,
                                           &overlaps,
                                           numberOfThreads](const int t,
                                                            const uint64_t labelId){
            auto & ovlp = overlaps[labelId];
            for(int tId = 0; tId < numberOfThreads; ++tId) {
                const auto & src = threadData[tId];
                const auto & srcIt = src.find(labelId);
                if(srcIt == src.end()) {
                    continue;
                }
                const auto & srcOvlps = srcIt->second;
                for(const auto & srcElem: srcOvlps) {
                    const uint64_t value = srcElem.first;
                    const uint64_t count = srcElem.second;

                    auto outIt = ovlp.find(value);
                    if(outIt == ovlp.end()) {
                        outIt = ovlp.emplace(std::make_pair(value, 0)).first;
                    }
                    outIt->second += count;
                }
            }
        });

        // serialzie the result
        // TODO implement the case max_overlap == False

        // find the maximum overlap value for each label
        xt::xtensor<uint64_t, 1> out = xt::zeros<uint64_t>({nLabels});
        nifty::parallel::parallel_foreach(numberOfThreads,
                                          nLabels,
                                          [&out,
                                           &overlaps](const int t,
                                                      const uint64_t labelId){
            const auto & ovlp = overlaps[labelId];
            uint64_t maxOvlp = 0;
            uint64_t maxOvlpValue = 0;
            for(const auto & elem: ovlp) {
                if(elem.second > maxOvlp) {
                    maxOvlp = elem.second;
                    maxOvlpValue = elem.first;
                }
            }
            out[labelId] = maxOvlpValue;
        });

        std::vector<std::size_t> outShape = {nLabels};
        std::vector<std::size_t> chunkShape = {std::min(nLabels, 262144UL)};
        auto dsOut = z5::createDataset(outputPath, "uint64",
                                       outShape, chunkShape, false,
                                       "gzip");

        const std::vector<size_t> zero1Coord({0});
        z5::multiarray::writeSubarray<uint64_t>(dsOut, out, zero1Coord.begin(), numberOfThreads);
    }


    template<class EDGES, class NODES>
    void connectedComponents(const Graph & graph,
                             const xt::xexpression<EDGES> & edges_exp,
                             const bool ignoreLabel,
                             xt::xexpression<NODES> & labels_exp) {
        const auto & edges = edges_exp.derived_cast();
        auto & labels = labels_exp.derived_cast();

        std::vector<NodeType> nodes;
        graph.nodes(nodes);

        // we need the number of nodes if nodes were dense
        const size_t nNodes = graph.nodeMaxId() + 1;

        // make union find
        std::vector<NodeType> rank(nNodes);
        std::vector<NodeType> parent(nNodes);
        boost::disjoint_sets<NodeType*, NodeType*> sets(&rank[0], &parent[0]);
        for(NodeType node_id = 0; node_id < nNodes; ++node_id) {
            sets.make_set(node_id);
        }

        // First pass:
        // iterate over each node and create new label at node
        // or assign representative of the neighbor node
        NodeType currentLabel = 0;
        for(const NodeType node : nodes){

            if(ignoreLabel && (node == 0)) {
                continue;
            }

            // iterate over the nodes in the neighborhood
            // and collect the nodes that are connected
            const auto & nhood = graph.nodeAdjacency(node);
            std::set<NodeType> ngbLabels;
            for(auto nhIt = nhood.begin(); nhIt != nhood.end(); ++nhIt) {
                const NodeType nhNode = nhIt->first;
                const EdgeIndexType nhEdge = nhIt->second;

                // nodes are connected if the edge has the value 0
                // this is in accordance with cut edges being 1
                if(!edges(nhEdge)) {
                    ngbLabels.insert(nhNode);
                }
            }

            // check if we are connected to any of the neighbors
            // and if the neighbor labels need to be merged
            if(ngbLabels.size() == 0) {
                // no connection -> make new label @ current pixel
                labels(node) = ++currentLabel;
            } else if (ngbLabels.size() == 1) {
                // only single label -> we assign its representative to the current pixel
                labels(node) = sets.find_set(*ngbLabels.begin());
            } else {
                // multiple labels -> we merge them and assign representative to the current pixel
                std::vector<NodeType> tmp_labels(ngbLabels.begin(), ngbLabels.end());
                for(unsigned ii = 1; ii < tmp_labels.size(); ++ii) {
                    sets.link(tmp_labels[ii - 1], tmp_labels[ii]);
                }
                labels(node) = sets.find_set(tmp_labels[0]);
            }
        }

        // Second pass:
        // Assign representative to each pixel
        for(const NodeType node : nodes){
            labels(node) = sets.find_set(labels(node));
        }
    }


    template<class OUT, class THREAD_DATA>
    inline void mergeBlockMapping(const THREAD_DATA & perThreadData,
                                  OUT & mapping, const int numberOfThreads) {
        // merge the label data into output vector
        const std::size_t numberOfLabels = mapping.size();
        nifty::parallel::parallel_foreach(numberOfThreads, numberOfLabels,
                                          [&](const int t,
                                              const uint64_t labelId){
            auto & out = mapping[labelId];
            for(int threadId = 0; threadId < numberOfThreads; ++threadId) {
                auto & threadData = perThreadData[threadId];
                auto it = threadData.find(labelId);
                if(it != threadData.end()) {
                    const auto & copyIds = it->second;
                    out.reserve(out.size() + copyIds.size());
                    out.insert(out.end(), copyIds.begin(), copyIds.end());
                }
            }
            std::sort(out.begin(), out.end());
        });

    }


    template<class OUT>
    inline void getBlockMapping(const std::string & inputPath,
                                const int numberOfThreads,
                                OUT & mapping) {

        auto inputDs = z5::openDataset(inputPath);

        // we store the mapping of labels to blocks extracted for each thread in an unordered map
        // of vectors
        typedef std::unordered_map<uint64_t, std::vector<std::size_t>> PerThread;
        std::vector<PerThread> perThreadData(numberOfThreads);

        const auto & blocking = inputDs->chunking();
        z5::util::parallel_for_each_chunk(*inputDs,
                                          numberOfThreads,
                                          [&perThreadData,
                                           &blocking](const int tId,
                                                      const z5::Dataset & ds,
                                                      const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            z5::handle::Chunk chunk(ds.handle(), chunkCoord, ds.isZarr());
            if(!chunk.exists()) {
                return;
            }

            bool isVarlen;
            const std::size_t chunkSize = ds.getDiscChunkSize(chunkCoord, isVarlen);
            std::vector<uint64_t> labelsInChunk(chunkSize);
            ds.readChunk(chunkCoord, &labelsInChunk[0]);

            // get the (1d) id of the chunk
            const std::size_t chunkId = blocking.blockCoordinatesToBlockId(chunkCoord);

            // add the chunkId to all the labels we have found in this chunk
            auto & threadData = perThreadData[tId];
            for(const uint64_t labelId: labelsInChunk) {
                auto it = threadData.find(labelId);
                if(it == threadData.end()) {
                    threadData.insert(std::make_pair(labelId,
                                                     std::vector<size_t>({chunkId})));
                } else {
                    it->second.push_back(chunkId);
                }
            }
        });

        mergeBlockMapping(perThreadData, mapping, numberOfThreads);
    }


    template<class OUT>
    inline void getBlockMappingWithRoi(const std::string & inputPath,
                                       const int numberOfThreads,
                                       OUT & mapping,
                                       const std::vector<std::size_t> & roiBegin,
                                       const std::vector<std::size_t> & roiEnd) {

        auto inputDs = z5::openDataset(inputPath);

        // we store the mapping of labels to blocks extracted for each thread in an unordered map
        // of vectors
        typedef std::unordered_map<uint64_t, std::vector<std::size_t>> PerThread;
        std::vector<PerThread> perThreadData(numberOfThreads);

        const auto & blocking = inputDs->chunking();
        z5::util::parallel_for_each_chunk_in_roi(*inputDs,
                                                 roiBegin,
                                                 roiEnd,
                                                 numberOfThreads,
                                                 [&perThreadData,
                                                  &blocking](const int tId,
                                                             const z5::Dataset & ds,
                                                             const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            z5::handle::Chunk chunk(ds.handle(), chunkCoord, ds.isZarr());
            if(!chunk.exists()) {
                return;
            }

            bool isVarlen;
            const std::size_t chunkSize = ds.getDiscChunkSize(chunkCoord, isVarlen);
            std::vector<uint64_t> labelsInChunk(chunkSize);
            ds.readChunk(chunkCoord, &labelsInChunk[0]);

            // get the (1d) id of the chunk
            const std::size_t chunkId = blocking.blockCoordinatesToBlockId(chunkCoord);

            // add the chunkId to all the labels we have found in this chunk
            auto & threadData = perThreadData[tId];
            for(const uint64_t labelId: labelsInChunk) {
                auto it = threadData.find(labelId);
                if(it == threadData.end()) {
                    threadData.insert(std::make_pair(labelId,
                                                     std::vector<size_t>({chunkId})));
                } else {
                    it->second.push_back(chunkId);
                }
            }
        });

        mergeBlockMapping(perThreadData, mapping, numberOfThreads);
    }


    template<class OUT>
    inline void serializeMappingChunks(const std::string & inputPath,
                                       const std::string & outputPath,
                                       const OUT & mapping,
                                       const int numberOfThreads) {
        // open the input and output datasets
        auto inputDs = z5::openDataset(inputPath);
        auto outputDs = z5::openDataset(outputPath);
        const auto & blocking = inputDs->chunking();

        const std::size_t numberOfLabels = mapping.size();
        const std::vector<std::size_t> idRoiBegin = {0};
        const std::vector<std::size_t> idRoiEnd = {numberOfLabels};
        z5::util::parallel_for_each_chunk_in_roi(*outputDs,
                                                 idRoiBegin,
                                                 idRoiEnd,
                                                 numberOfThreads,
                                                 [&mapping,
                                                  &blocking,
                                                  numberOfLabels](const int t,
                                                                  const z5::Dataset & ds,
                                                                  const z5::types::ShapeType & idChunk){

            const auto & idBlocking = ds.chunking();
            // get the begin and end in id-space for this chunk
            std::vector<std::size_t> idBegin, idEnd;
            idBlocking.getBlockBeginAndEnd(idChunk, idBegin, idEnd);
            idEnd[0] = std::min(idEnd[0], numberOfLabels);

            // calculate the serialization size for this chunk in byte
            std::size_t serSize = 0;
            for(int64_t labelId = idBegin[0]; labelId < idEnd[0]; ++labelId) {
                const std::size_t nBlocks = mapping[labelId].size();
                // for every label that is present in at least one block, we serialize:
                // labelId as int64 = 8 byte
                // number of blocks as int32 = 4 byte
                // 6 int64 coordinates for each block = 6 * 8 * nBlocks = 48 * nBlocks
                if(nBlocks > 0) {
                    serSize += 12 + nBlocks * 48;
                }
            }

            std::vector<size_t> chunkBegin, chunkEnd;
            // make serialzation
            char * byteSerialization = new char[serSize];
            char * serPointer = byteSerialization;
            for(int64_t labelId = idBegin[0]; labelId < idEnd[0]; ++labelId) {
                const auto & blockList = mapping[labelId];
                int32_t nBlocks = static_cast<int32_t>(blockList.size());
                if(nBlocks > 0) {
                    // copy labelId, numberOfBlocks into the serialization buffer
                    int64_t labelIdOut = labelId;

                    // for some reason, this is not in the n5 default endianness,
                    // so we need to reverse the endidianness for everything here
                    z5::util::reverseEndiannessInplace(labelIdOut);
                    memcpy(serPointer, &labelIdOut, 8);
                    serPointer += 8;

                    z5::util::reverseEndiannessInplace(nBlocks);
                    memcpy(serPointer, &nBlocks, 4);
                    serPointer += 4;

                    // serialize the coordinates for all blocks in blocklist
                    for(const std::size_t chunkId: blockList) {
                        blocking.getBlockBeginAndEnd(chunkId, chunkBegin, chunkEnd);

                        // NOTE, java has axis order XYZ, we have ZYX that's why we revert
                        // also, we report the end coordinates (= max + 1), java expects max
                        std::array<int64_t, 6> blockSer = {static_cast<int64_t>(chunkBegin[2]),
                                                           static_cast<int64_t>(chunkBegin[1]),
                                                           static_cast<int64_t>(chunkBegin[0]),
                                                           static_cast<int64_t>(chunkEnd[2] - 1),
                                                           static_cast<int64_t>(chunkEnd[1] - 1),
                                                           static_cast<int64_t>(chunkEnd[0] - 1)};

                        for(int64_t bc : blockSer) {
                            z5::util::reverseEndiannessInplace(bc);
                            memcpy(serPointer, &bc, 8);
                            serPointer += 8;
                        }
                    }
                }
            }
            // write serialization to this chunk
            ds.writeChunk(idChunk, byteSerialization, true, serSize);
            delete[] byteSerialization;
        });

    }


    inline void serializeBlockMapping(const std::string & inputPath,
                                      const std::string & outputPath,
                                      const std::size_t numberOfLabels,
                                      const int numberOfThreads,
                                      const std::vector<size_t> & roiBegin=std::vector<std::size_t>(),
                                      const std::vector<size_t> & roiEnd=std::vector<std::size_t>()) {

        // iterate over the input in parallel and map block-ids to label ids
        std::vector<std::vector<std::size_t>> mapping(numberOfLabels);
        if(roiBegin.size() == 0) {
            getBlockMapping(inputPath, numberOfThreads, mapping);
        } else if(roiBegin.size() == 3) {
            getBlockMappingWithRoi(inputPath, numberOfThreads, mapping, roiBegin, roiEnd);
        } else {
            throw std::runtime_error("Invalid ROI");
        }
        serializeMappingChunks(inputPath, outputPath, mapping, numberOfThreads);
    }


    // take block mapping serialization and return the map
    inline void formatBlockMapping(const std::vector<char> & input,
                                   std::map<std::uint64_t, std::vector<std::array<int64_t, 6>>> & mapping) {
        const std::size_t byteSize = input.size();
        const char * serPointer = &input[0];

        int64_t labelId;
        int32_t nBlocks;
        std::array<int64_t, 6> coords;
        while(std::distance(&input[0], serPointer) < byteSize) {
            memcpy(&labelId, serPointer, 8);
            z5::util::reverseEndiannessInplace(labelId);
            serPointer += 8;

            memcpy(&nBlocks, serPointer, 4);
            z5::util::reverseEndiannessInplace(nBlocks);
            serPointer += 4;

            std::vector<std::array<int64_t, 6>> coordList;
            for(int i = 0; i < nBlocks; ++i) {
                for(int c = 0; c < 6; ++c) {
                    memcpy(&coords[c], serPointer, 8);
                    serPointer += 8;
                }
                z5::util::reverseEndiannessInplace<int64_t>(coords.begin(), coords.end());
                coordList.push_back(coords);
            }
            mapping[static_cast<uint64_t>(labelId)] = coordList;
        }
    }


    inline void readBlockMapping(const std::string & dsPath,
                                 const std::vector<std::size_t> chunkId,
                                 std::map<std::uint64_t, std::vector<std::array<int64_t, 6>>> & mapping) {
        auto ds = z5::openDataset(dsPath);
        if(ds->chunkExists(chunkId)) {
            bool isVarlen;
            const std::size_t chunkSize = ds->getDiscChunkSize(chunkId, isVarlen);
            std::vector<char> out(chunkSize);
            ds->readChunk(chunkId, &out[0]);
            formatBlockMapping(out, mapping);
        }
    }

}
}
