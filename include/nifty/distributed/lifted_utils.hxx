#pragma once

#include "z5/util/for_each.hxx"
#include "z5/util/util.hxx"

#include "nifty/tools/blocking.hxx"
#include "nifty/distributed/graph_extraction.hxx"
#include "nifty/distributed/distributed_graph.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {


    inline void liftedEdgesFromNode(const Graph & graph,
                                    const uint64_t srcNode,
                                    const unsigned graphDepth,
                                    std::vector<EdgeType> & out) {
        // type to put on bfs queue, stores the node id and the graph
        // distance of this node from the start node
        typedef std::pair<uint64_t, unsigned> QueueElem;

        // set of visited nodes
        std::unordered_set<uint64_t> visited;
        // queue of nodes to visit
        std::queue<QueueElem> queue;

        queue.emplace(std::make_pair(srcNode, 0));

        while(queue.size()) {
            const QueueElem elem = queue.front();
            queue.pop();

            const uint64_t node = elem.first;
            // check if this node was visited already
            if(visited.find(node) != visited.end()) {
                continue;
            }
            visited.insert(node);

            // increase depth and check if we continue search from this node
            unsigned depth = elem.second;
            if(depth > graphDepth) {
                continue;
            }

            // put the neighboring nodes on the queue
            // (only if we wouldn't exceed max node depth )
            if(depth < graphDepth) {
                const auto & adj = graph.nodeAdjacency(node);
                for(const auto & ngb: adj) {
                    // put node on queue if it has not been visited already
                    const uint64_t ngbNode = ngb.first;
                    if(visited.find(ngbNode) == visited.end()) {
                        queue.emplace(std::make_pair(ngbNode, depth + 1));
                    }
                }
            }

            // check if we make a lifted edge between the start node and this node:
            // is this a lifted edge? i.e. graph depth > 1
            const bool isLiftedEdge = depth > 1;
            if(isLiftedEdge) {
                out.emplace_back(std::make_pair(srcNode, node));
            }
        }
    }


    template<class NODE_LABELS, class F>
    inline void findLiftedEdgesBfs(const Graph & graph,
                                   const uint64_t srcNode,
                                   const NODE_LABELS & nodeLabels,
                                   const unsigned graphDepth,
                                   std::vector<EdgeType> & out,
                                   F && addEdge,
                                   const uint64_t ignoreLabel=0) {
        // type to put on bfs queue, stores the node id and the graph
        // distance of this node from the start node
        typedef std::pair<uint64_t, unsigned> QueueElem;

        // set of visited nodes
        std::unordered_set<uint64_t> visited;
        // queue of nodes to visit
        std::queue<QueueElem> queue;

        queue.emplace(std::make_pair(srcNode, 0));
        const uint64_t srcLabel = nodeLabels[srcNode];

        while(queue.size()) {
            const QueueElem elem = queue.front();
            queue.pop();

            const uint64_t node = elem.first;
            // check if this node was visited already
            if(visited.find(node) != visited.end()) {
                continue;
            }
            visited.insert(node);

            // increase depth and check if we continue search from this node
            unsigned depth = elem.second;
            if(depth > graphDepth) {
                continue;
            }

            // put the neighboring nodes on the queue
            // (only if we wouldn't exceed max node depth )
            if(depth < graphDepth) {
                // FIXME map can get out of scope, I don't understand why
                try {
                    const auto & adj = graph.nodeAdjacency(node);
                    for(const auto & ngb: adj) {
                        // put node on queue if it has not been visited already
                        const uint64_t ngbNode = ngb.first;
                        if(visited.find(ngbNode) == visited.end()) {
                            queue.emplace(std::make_pair(ngbNode, depth + 1));
                        }
                    }
                } catch(std::out_of_range exception) {
                    continue;
                }
            }

            // check if we make a lifted edge between the start node and this node:
            // 1.) is this a lifted edge? i.e. graph depth > 1
            const bool isLiftedEdge = depth > 1;
            if(depth <= 1) {
                continue;
            }

            // 2.) does the node have a label?
            const uint64_t label = nodeLabels[node];
            if(label == ignoreLabel) {
                continue;
            }

            // 3.) is srcNode < node? (make sure not to duplicate lifted edges)
            if(srcNode > node) {
                continue;
            }

            // 4) additional lifted check dependend on the node labels
            if(!addEdge(srcLabel, label)) {
                continue;
            }

            // all checks passed ? -> add the lifted edges
            out.emplace_back(std::make_pair(srcNode, node));
        }
    }


    inline bool addAll(const uint64_t labelA, const uint64_t labelB) {
        return true;
    }


    inline bool addSame(const uint64_t labelA, const uint64_t labelB) {
        return labelA == labelB;
    }


    inline bool addDifferent(const uint64_t labelA, const uint64_t labelB) {
        return labelA != labelB;
    }


    inline void computeLiftedNeighborhoodFromNodeLabels(const std::string & graphPath,
                                                        const std::string & nodeLabelPath,
                                                        const std::string & outputPath,
                                                        const unsigned graphDepth,
                                                        const int numberOfThreads,
                                                        const std::string & mode="all",
                                                        const uint64_t ignoreLabel=0) {
        // modes can be
        // "all": add all edges between nodes with a label
        // "same": add edges only between nodes with the same label
        // "different": add edges only between nodes with different labels
        auto edgeChecker = (mode=="all") ? addAll : ((mode=="same") ? addSame : addDifferent);

        // load the graph
        const auto graph = Graph(graphPath, numberOfThreads);

        // load the node labels
        auto nodeDs = z5::openDataset(nodeLabelPath);
        const std::size_t nNodes = nodeDs->shape(0);
        const std::vector<std::size_t> zero1Coord({0});
        Shape1Type nodeShape({nNodes});
        Tensor1 nodeLabels(nodeShape);
        z5::multiarray::readSubarray<uint64_t>(nodeDs, nodeLabels, zero1Coord.begin());

        // per thread data: store lifted edges for each thread
        std::vector<std::vector<EdgeType>> perThreadData(numberOfThreads);

        // find lifted edges via bfs starting from each node. (in parallel)
        // only add edges if both nodes have a node label
        nifty::parallel::parallel_foreach(numberOfThreads, nNodes,
                                          [&](const int tid, const uint64_t nodeId){
            // zero is usually not in graph
            // TODO should do a proper check instead ...
            if(nodeId == 0) {
                return;
            }

            // check if this node has a node label
            const uint64_t nodeLabel = nodeLabels[nodeId];
            // continue if it is unlabeled
            if(nodeLabel == ignoreLabel) {
                return;
            }
            auto & threadData = perThreadData[tid];
            // do bfs for this node and find all relevant lifted edges
            findLiftedEdgesBfs(graph, nodeId,
                               nodeLabels, graphDepth, threadData,
                               edgeChecker, ignoreLabel);
        });

        // merge the thread
        std::size_t nLifted = 0;
        for(int tid = 0; tid < numberOfThreads; ++tid) {
            nLifted += perThreadData[tid].size();
        }
        auto & liftedEdges = perThreadData[0];
        liftedEdges.reserve(nLifted);
        for(int tid = 1; tid < numberOfThreads; ++tid) {
            const auto & src = perThreadData[tid];
            liftedEdges.insert(liftedEdges.end(), src.begin(), src.end());
        }
        // sort lifted edges by node ids
        std::sort(liftedEdges.begin(), liftedEdges.end());

        // serialize
        Shape2Type outShape = {nLifted, 2};
        xt::xtensor<uint64_t, 2> out(outShape);

        for(std::size_t liftedId = 0; liftedId < nLifted; ++liftedId) {
            const auto & lifted = liftedEdges[liftedId];
            out(liftedId, 0) = lifted.first;
            out(liftedId, 1) = lifted.second;
        }

        std::vector<std::size_t> dsShape = {nLifted, 2};
        std::vector<std::size_t> dsChunks = {std::min(static_cast<std::size_t>(64*64*64), nLifted), 1};
        auto dsOut = z5::createDataset(outputPath, "uint64",
                                       dsShape, dsChunks, false,
                                       "gzip");
        const std::vector<std::size_t> zero2Coord = {0, 0};
        z5::multiarray::writeSubarray<uint64_t>(dsOut, out, zero2Coord.begin(), numberOfThreads);
    }

}
}
