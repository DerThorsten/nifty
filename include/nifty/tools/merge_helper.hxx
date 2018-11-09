#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include <boost/functional/hash.hpp>

#include "nifty/xtensor/xtensor.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty {
namespace tools {

    template<class UV_ARRAY, class INDICATORS, class SIZES>
    void computeMergeVotes(const xt::xexpression<UV_ARRAY> & uvIdsExp,
                           const xt::xexpression<INDICATORS> & indicatorsExp,
                           const xt::xexpression<SIZES> & sizesExp,
                           std::map<std::pair<typename UV_ARRAY::value_type,
                                              typename UV_ARRAY::value_type>,
                                    std::pair<size_t, size_t>> & mergeVotes,
                           const bool weightEdges=false) {
        const auto & uvIds = uvIdsExp.derived_cast();
        const auto & indicators = indicatorsExp.derived_cast();
        const auto & sizes = sizesExp.derived_cast();

        typedef typename UV_ARRAY::value_type NodeType;
        typedef typename std::pair<NodeType, NodeType> UvType;
        mergeVotes.clear();

        // TODO parallelize ?!
        const size_t nPairs = uvIds.shape()[0];
        for(size_t i = 0; i < nPairs; ++i) {
            const NodeType u = uvIds(i, 0);
            const NodeType v = uvIds(i, 1);
            const UvType uv(u, v);

            auto voteIt = mergeVotes.find(uv);
            if(voteIt == mergeVotes.end()) {
                voteIt = mergeVotes.insert(voteIt, std::make_pair(uv,
                                                                  std::make_pair(0, 0)));
            }

            const auto indicator = indicators(i);
            if(weightEdges) {
                auto size = sizes(i);
                voteIt->second.first += indicator * size;
                voteIt->second.second += size;
            } else {
                voteIt->second.first += indicator;
                voteIt->second.second += 1;
            }

        }
    }


    template<class UV_ARRAY, class VOTES>
    void mergeMergeVotes(const xt::xexpression<UV_ARRAY> & uvIdsExp,
                         const xt::xexpression<VOTES> & votesExp,
                         std::map<std::pair<typename UV_ARRAY::value_type,
                                            typename UV_ARRAY::value_type>,
                                  std::pair<size_t, size_t>> & votesOut) {

        const auto & uvIds = uvIdsExp.derived_cast();
        const auto & votes = votesExp.derived_cast();

        typedef typename UV_ARRAY::value_type NodeType;
        typedef typename std::pair<NodeType, NodeType> UvType;
        votesOut.clear();

        // TODO parallelize ?!
        const size_t nPairs = uvIds.shape()[0];
        for(size_t i = 0; i < nPairs; ++i) {
            const NodeType u = uvIds(i, 0);
            const NodeType v = uvIds(i, 1);
            const UvType uv(u, v);

            // find this node pair in the out dataset
            auto voteIt = votesOut.find(uv);
            if(voteIt == votesOut.end()) {
                voteIt = votesOut.insert(voteIt, std::make_pair(uv,
                                                                std::make_pair(0, 0)));
            }

            // add the current votes
            voteIt->second.first += votes(i, 0);
            voteIt->second.second += votes(i, 1);
        }

    }


}
}
