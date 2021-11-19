//
// Created by Alberto Bailoni on 07/01/2021.
//

#pragma once

#include <functional>
#include <set>
#include <vector>
#include <unordered_set>
#include <boost/container/flat_set.hpp>
#include <string>
#include <cmath>        // std::abs

// #include "nifty/tools/changable_priority_queue.hxx"
// #include "nifty/graph/edge_contraction_graph.hxx"
// #include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include <iostream>
#include "xtensor/xarray.hpp"

#include "nifty/parallel/threadpool.hxx"
#include <ctime> // Random seed for shuffle
#include <xtensor/xrandom.hpp>


namespace nifty{
    namespace graph {
        template<class GRAPH, class LABELS, class SIGNED_WEIGHTS, class IS_LOCAL_EDGE>
        void runLabelPropagation(const GRAPH & graph,
                                 LABELS & nodeLabels,
                                 const SIGNED_WEIGHTS & signedWeights,
                                 const IS_LOCAL_EDGE & localEdges,
                                 const uint64_t nb_iter=1,
                                 const int64_t size_constr=-1,
                                 const int64_t nbThreads=-1){
            typedef std::set<uint64_t> uintSetType;
            const auto nbNodes = graph.nodeIdUpperBound()+1;
            xt::xtensor<uint64_t, 1> clusterSizes = xt::ones<uint64_t>({nbNodes});
            xt::xtensor<uint64_t, 1> shuffled_nodes = xt::arange<uint64_t>(0, nbNodes, 1);
            xt::random::seed(time(NULL));
            // Shuffle the order of the nodes:
            auto engine = xt::random::get_default_random_engine();

            const parallel::ParallelOptions pOpts(nbThreads);
            const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

            xt::xtensor<double, 2> neighValues = xt::zeros<double>({actualNumberOfThreads, (std::size_t) graph.nodeIdUpperBound()+1});
            // xt::xtensor<double, 1> neighSizes = xt::zeros<double>({actualNumberOfThreads, (std::size_t) graph.nodeIdUpperBound()+1});
            xt::xtensor<bool, 2> neighLocality = xt::zeros<bool>({actualNumberOfThreads, (std::size_t) graph.nodeIdUpperBound()+1});

            for (uint64_t iter = 0; iter < nb_iter; ++iter) {
                xt::random::shuffle(shuffled_nodes, engine);

                parallel::ThreadPool threadpool(pOpts);
                nifty::parallel::parallel_foreach(threadpool,
                                           shuffled_nodes.begin(),
                                           shuffled_nodes.end(),
                                           [&](const int tid, const uint64_t node){
                                               const auto oldLabel = nodeLabels(node);
                                               uintSetType neighStats;

                                               for (auto adj : graph.adjacency(node)) {
                                                   const auto neighNode = adj.node();
                                                   const auto neighEdge = adj.edge();
                                                   const auto neighLabel = nodeLabels(neighNode);
                                                   auto neighSize = clusterSizes(neighLabel);
                                                   if (neighLabel == oldLabel) {
                                                       neighSize--;
                                                   }
                                                   if (size_constr > 0) {
                                                       if (neighSize >= size_constr) {
                                                           continue;
                                                       }
                                                   }
                                                   // Update stats:
                                                   neighStats.insert(neighLabel);
                                                   neighValues(tid, neighLabel) = neighValues(tid, neighLabel) + signedWeights(neighEdge);
                                                   // neighSizes(neighLabel) = neighSizes(neighLabel) + 1;
                                                   auto edge_is_local = localEdges(neighEdge);
                                                   neighLocality(tid, neighLabel) = neighLocality(tid, neighLabel) || edge_is_local;
                                               }
                                               // Find max label:
                                               uintSetType maxLabels;
                                               // TODO: update and add new label if all are repulsive!
                                               maxLabels.insert(oldLabel);
                                               double max = 0.;
                                               for (auto neighborCluster : neighStats) {
                                                   if (neighLocality(tid, neighborCluster)) {
                                                       auto prio = neighValues(tid, neighborCluster);
                                                       if (prio > max) {
                                                           max = prio;
                                                           maxLabels = uintSetType();
                                                           maxLabels.insert(neighborCluster);
                                                       } else if (prio == max) {
                                                           maxLabels.insert(neighborCluster);
                                                       }
                                                   }
                                                   // Reset stats for next node:
                                                   neighValues(tid, neighborCluster) = 0;
                                                   neighLocality(tid, neighborCluster) = false;
                                               }

                                               // If more than one, extract one randomly:
                                               uint64_t selectedCluster;
                                               if (maxLabels.size() > 1) {
                                                   const uint64_t nb_labels = maxLabels.size();
                                                   xt::xtensor<uint64_t, 1> randomLabel = xt::random::randint({1}, (uint64_t) 0, nb_labels, engine);
                                                   selectedCluster = *std::next(maxLabels.begin(), randomLabel(0));
                                               } else {
                                                   selectedCluster = *maxLabels.begin();
                                               }

                                               nodeLabels(node) = selectedCluster;
                                               clusterSizes(oldLabel)--;
                                               clusterSizes(selectedCluster)++;
                                           });

                // for (const uint64_t node : shuffled_nodes) {
                //
                // }
            }
        }
    }
}
