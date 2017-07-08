#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "nifty/marray/marray.hxx"
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

    EdgeMapping(const size_t numberOfEdges, const int nThreads=-1)
        : oldToNewEdges_(numberOfEdges), newUvIds_(), threadpool_(nThreads)
    {}

    void initializeMapping(const marray::View<EdgeType> & uvIds, const std::vector<NodeType> & oldToNewNodes);

    template<class T>
    void mapEdgeValues(const std::vector<T> & edgeValues, std::vector<T> & newEdgeValues) const;

    const UvVectorType & getNewUvIds() const
    {return newUvIds_;}

    void getNewEdgeIds(const std::vector<EdgeType> & edgeIds, std::vector<EdgeType> & newEdgeIds) const;

    size_t numberOfNewEdges() const
    {return newUvIds_.size();}

private:

    //std::map<EdgeType, std::vector<EdgeType>> newToOldEdges_;
    std::vector<EdgeType> oldToNewEdges_;
    UvVectorType newUvIds_;
    mutable parallel::ThreadPool threadpool_;
};


// TODO parallelize ?!
template<class EDGE_TYPE, class NODE_TYPE>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::initializeMapping(const marray::View<EdgeType> & uvIds, const std::vector<NodeType> & oldToNewNodes)
{
    //NodeType uNew, vNew;
    //UvType uvNew;
    //std::map<UvType, EdgeType> indexMap;
    //EdgeType index = 0;

    //for(EdgeType i = 0; i < uvIds.shape(0); ++i) {
    //    uNew = oldToNewNodes[uvIds(i,0)];
    //    vNew = oldToNewNodes[uvIds(i,1)];

    //    if(uNew == vNew) {
    //        oldToNewEdges_[i] = -1;
    //        continue;
    //    }

    //    uvNew = std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew));

    //    auto e = indexMap.insert(std::make_pair(uvNew, index));
    //    if(e.second) {
    //        newUvIds_.push_back(uvNew);
    //        ++index;
    //    }

    //    oldToNewEdges_[i] = e.first->second;
    //}

    std::vector< std::map<UvType, std::vector<EdgeType>> > threadMaps(threadpool_.nThreads());

    // find new uv ids in parallel
    parallel::parallel_foreach(threadpool_, uvIds.shape(0), [&](const int tId, const EdgeType edgeId){
        NodeType uNew, vNew = oldToNewNodes[uvIds(edgeId, 0)], oldToNewNodes[uvIds(edgeId, 1)];

        if(uNew == vNew) {
            //NOTE this is threadsafe, because we are looping over edgeId in parralel
            oldToNewEdges_[edgeId] = -1;
            return;
        }

        // perform map insertion with hint
        auto & thisMap = threadMaps[tId];
        auto uvNew = std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew));
        auto mapIt = thisMap.lower_bound(uvNew);
        // key is already in map
        if( mapIt != thisMap.end() && !(thisMap.key_comp()(uvNew, mapIt->first)) ) {
            mapIt->second.push_back(edgeId);
        }
        else { // key is not in map
            //thisMap.insert(mapIt, std::make_pair(uvNew, std::vector<EdgeType>({edgeId})) );
            thisMap.emplace_hint(mapIt, uvNew, std::vector<EdgeType>({edgeId}));
        }

    });

    // find the unique new uv-ids with a set, then copy them in a vector to have
    // constant time access
    std::set<UvType> uniqueKeysTmp;
    for(size_t t = 0; t < threadpool_.nThreads(); ++t) {
        const auto & thisMap = threadMaps[t];
        for(auto mapIt = thisMap.begin(); mapIt != thisMap.end(); ++mapIt) {
            uniqueKeysTmp.insert(mapIt->first);
        }
    }
    std::vector<UvType> uniqueKeys(uniqueKeysTmp.size());
    std::copy(uniqueKeysTmp.begin(), uniqueKeysTmp.end(), uniqueKeys.begin());

    // find old to new edges in parallel by going over the unique new uv-ids
    parallel::parallel_foreach(threadpool_, uniqueKeys.size(), [&](const int tId, const int newEdgeId){

        // acces the uv id in the set by index
        //
        const auto & uvNew = uniqueKeys[newEdgeId];

        for(int t = 0; t < threadpool_.nThreads(); ++t) {
            const auto & thisMap = threadMaps[t];
            auto mapIt = thisMap.find(uvNew);
            if(mapIt != thisMap.end()) {
                for(auto oldEdgeId : mapIt->second) {
                    oldToNewEdges_[oldEdgeId] = newEdgeId;
                }
            }
        }
    });

}


template<class EDGE_TYPE, class NODE_TYPE>
template<class T>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::mapEdgeValues(
        const std::vector<T> & edgeValues, std::vector<T> & newEdgeValues) const {

    NIFTY_CHECK_OP(edgeValues.size(),==,oldToNewEdges_.size(),"Wrong Input size");

    newEdgeValues.clear();
    newEdgeValues.resize(newUvIds_.size(), 0);

    // initialise the thread data
    std::vector<std::vector<T>> threadVecs(threadpool_.nThreads());
    parallel::parallel_foreach(threadpool_, threadpool_.nThreads(), [&](const int t, const int i){
        threadVecs[i] = std::vector<T>(newUvIds_.size(), 0);
    });

    // extract the new edge values in parallel over the old edges
    parallel::parallel_foreach(threadpool_, edgeValues.size(), [&](const int tId, const int edgeId){
        auto newEdge = oldToNewEdges_[edgeId];
        if(newEdge != -1) {
            auto & newVals = threadVecs[tId];
            newVals[newEdge] += edgeValues[edgeId];
        }
    });

    // write to the out vector in parallel over the new edges
    parallel::parallel_foreach(threadpool_, newEdgeValues.size(), [&](const int tId, const int newEdgeId){
        auto & destValue = newEdgeValues[newEdgeId];
        for(int t = 0; t < threadpool_.nThreads(); ++t) {
            auto & srcValues = threadVecs[t];
            destValue += srcValues[newEdgeId];
        }
    });

}


template<class EDGE_TYPE, class NODE_TYPE>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::getNewEdgeIds(
        const std::vector<EdgeType> & edgeIds, std::vector<EdgeType> & newEdgeIds) const {

    newEdgeIds.clear();

    std::vector<std::set<EdgeType>> threadSets(threadpool_.nThreads());

    // find new edges in parallel
    parallel::parallel_foreach(threadpool_, edgeIds.size(), [&](const int tId, const int edgeId){
        auto newEdge = oldToNewEdges_[edgeId];
        if(newEdge != -1) {
            auto & thisSet = threadSets[tId];
            thisSet.insert(newEdge);
        }
    });

    // merge the edge sets
    auto & destSet = threadSets[0];
    for(int tId = 1; tId < threadpool_.nThreads(); ++tId) {
        const auto & srcSet = threadSets[tId];
        destSet.insert(srcSet.begin(), srcSet.end());
    }

    // write edge values to out vector
    newEdgeIds.resize(destSet.size());
    std::copy(destSet.begin(), destSet.end(), newEdgeIds.begin());

}


}
}
