#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include <boost/functional/hash.hpp>

#include "nifty/xtensor/xtensor.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty {
namespace tools {

template<class EDGE_TYPE, class NODE_TYPE>
class EdgeMapping {

public:

    typedef EDGE_TYPE EdgeType;
    typedef NODE_TYPE NodeType;
    typedef std::pair<NodeType, NodeType> UvType;
    typedef std::vector<UvType> UvVectorType;

    template<class UV_ARRAY, class NODE_ARRAY>
    EdgeMapping(const xt::xexpression<UV_ARRAY> & uvIds,
                const xt::xexpression<NODE_ARRAY> & nodeLabeling,
                const int nThreads=-1)
    {
        initializeMapping(uvIds, nodeLabeling, nThreads);
    }

    template<class VALARRAY>
    void mapEdgeValues(const xt::xexpression<VALARRAY> &, xt::xexpression<VALARRAY> &, const int) const;

    const UvVectorType & newUvIds() const
    {return newUvIds_;}

    void getNewEdgeIds(const std::vector<EdgeType> &, std::vector<EdgeType> &) const;

    size_t numberOfNewEdges() const
    {return newUvIds_.size();}


private:

    template<class UV_ARRAY, class NODE_ARRAY>
    void initializeMapping(const xt::xexpression<UV_ARRAY> & uvIdsExp,
                           const xt::xexpression<NODE_ARRAY> & nodeLabelingExp,
                           const int numberOfThreads) {
        const auto & uvIds = uvIdsExp.derived_cast();
        const auto & nodeLabeling = nodeLabelingExp.derived_cast();

        const size_t nEdges = uvIds.shape()[0];
        nifty::parallel::ThreadPool threadpool(numberOfThreads);
        const size_t nThreads = threadpool.nThreads();

        // find new uv-ids
        typedef boost::hash<UvType> Hash;
        std::unordered_map<UvType, size_t, Hash> uvNewToIndex;
        {
            // use normal map because we want to have ordered keys
            typedef std::set<UvType> UvSet;
            std::vector<UvSet> perThreadData(threadpool.nThreads());

            nifty::parallel::parallel_foreach(threadpool, nEdges, [&](const int tId, const EdgeType edgeId) {
                const NodeType u = uvIds(edgeId, 0);
                const NodeType v = uvIds(edgeId, 1);
                const NodeType uNew = nodeLabeling(u);
                const NodeType vNew = nodeLabeling(v);
                if(uNew == vNew) {
                    return;
                }
                perThreadData[tId].insert(std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew)));
            });

            // merge
            auto & uvSet = perThreadData[0];
            for(int tId = 1; tId < nThreads; ++tId) {
                uvSet.insert(perThreadData[tId].begin(), perThreadData[tId].end());
            }

            newUvIds_.resize(uvSet.size());
            size_t ii = 0;
            for(const auto & uv: uvSet) {
                newUvIds_[ii] = uv;
                uvNewToIndex[uv] = ii;
                ++ii;
            }
        }

        // get the edge mapping
        {
            edgeMapping_.resize(nEdges);

            nifty::parallel::parallel_foreach(threadpool, nEdges, [&](const int tId, const EdgeType edgeId) {
                const NodeType uNew = nodeLabeling(edgeId, 0);
                const NodeType vNew = nodeLabeling(edgeId, 1);
                if(uNew == vNew) {
                    edgeMapping_[edgeId] = -1;
                    return;
                }
                const auto uvNew = std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew));
                const auto newEdgeId = uvNewToIndex[uvNew];
                edgeMapping_[edgeId] = newEdgeId;
            });
        }

    }

    std::vector<EdgeType> edgeMapping_;
    std::vector<UvType> newUvIds_;
};


// TODO different mapping functions
template<class EDGE_TYPE, class NODE_TYPE>
template<class VALARRAY>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::mapEdgeValues(const xt::xexpression<VALARRAY> & edgeValuesExp,
                                                      xt::xexpression<VALARRAY> & newEdgeValuesExp,
                                                      const int numberOfThreads) const {

    typedef typename VALARRAY::value_type ValueType;
    const auto & edgeValues = edgeValuesExp.derived_cast();
    auto & newEdgeValues = newEdgeValuesExp.derived_cast();

    NIFTY_CHECK_OP(edgeValues.shape()[0], ==, edgeMapping_.size(), "Wrong Input size");

    nifty::parallel::ThreadPool threadpool(numberOfThreads);
    const size_t nThreads = threadpool.nThreads();

    // initialise the thread data
    std::vector<std::vector<ValueType>> perThreadData(nThreads);
    parallel::parallel_foreach(threadpool, nThreads, [&](const int t, const int i){
        if(i != 0) {
            perThreadData[i] = std::vector<ValueType>(newUvIds_.size(), 0);
        }
    });

    // extract the new edge values in parallel over the old edges
    parallel::parallel_foreach(threadpool, edgeValues.size(), [&](const int tId, const int edgeId){
        const int64_t newEdge = edgeMapping_[edgeId];
        if(newEdge != -1) {
            ValueType * val = &newEdgeValues(newEdge);
            if(tId != 0) {
                val = &perThreadData[tId][newEdge];
            }
            *val += edgeValues(edgeId);
        }
    });

    // write to the out vector in parallel over the new edges
    parallel::parallel_foreach(threadpool, newEdgeValues.size(), [&](const int tId, const int newEdgeId){
        auto & val = newEdgeValues(newEdgeId);
        for(int t = 1; t < nThreads; ++t) {
            val += perThreadData[t][newEdgeId];
        }
    });

}


template<class EDGE_TYPE, class NODE_TYPE>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::getNewEdgeIds(const std::vector<EdgeType> & edgeIds,
                                                      std::vector<EdgeType> & newEdgeIds) const {

    newEdgeIds.clear();
    for(auto oldEdge : edgeIds) {
        const auto newEdge = edgeMapping_[oldEdge];
        if(newEdge != -1) {
            newEdgeIds.push_back(newEdge);
        }
    }

    std::sort(newEdgeIds.begin(), newEdgeIds.end());
    auto edgeIt = std::unique(newEdgeIds.begin(), newEdgeIds.end());
    newEdgeIds.erase(edgeIt, newEdgeIds.end());
}


}
}
