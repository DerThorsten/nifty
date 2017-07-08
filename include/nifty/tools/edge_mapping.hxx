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
    NodeType uNew, vNew;
    UvType uvNew;
    std::map<UvType, EdgeType> indexMap;
    EdgeType index = 0;

    for(EdgeType i = 0; i < uvIds.shape(0); ++i) {
        uNew = oldToNewNodes[uvIds(i,0)];
        vNew = oldToNewNodes[uvIds(i,1)];

        if(uNew == vNew) {
            oldToNewEdges_[i] = -1;
            continue;
        }

        uvNew = std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew));

        auto e = indexMap.insert(std::make_pair(uvNew, index));
        if(e.second) {
            newUvIds_.push_back(uvNew);
            ++index;
        }

        oldToNewEdges_[i] = e.first->second;
    }

    // FIXME there is some bug in parallelisation
    /*
    typedef std::vector< std::pair<UvType, EdgeType> > newUvToOldEdgeVector;
    std::vector<newUvToOldEdgeVector> threadVectors(threadpool_.nThreads());

    // find new uv ids in parallel
    parallel::parallel_foreach(threadpool_, uvIds.shape(0), [&](const int tId, const EdgeType edgeId){
        NodeType uNew, vNew = oldToNewNodes[uvIds(edgeId, 0)], oldToNewNodes[uvIds(edgeId, 1)];

        if(uNew == vNew) {
            //NOTE this is threadsafe, because we are looping over edgeId in parralel
            oldToNewEdges_[edgeId] = -1;
            return;
        }

        // perform map insertion with hint
        auto & thisVector = threadVectors[tId];
        auto uvNew = std::make_pair(std::min(uNew, vNew), std::max(uNew, vNew));
        thisVector.emplace_back( std::make_pair(uvNew, edgeId) );

    });

    // insert the uv-ids into newUvIds
    // surprisingly, initial benchmarks have shown that the 'find' approach 
    // is fastest for keeping the newUvIds unique.
    // TODO benchmark this again properly
    //for(size_t t = 0; t < threadpool_.nThreads(); ++t) {
    //    const auto & thisVector = threadVectors[t];
    //    for(const auto & elem : thisVector) {

    //        if(std::find(newUvIds_.begin(), newUvIds_.end(), elem.first) == newUvIds_.end()) {
    //            newUvIds_.push_back(elem.first);
    //        }
    //    }

    //}
    //std::sort(newUvIds_.begin(), newUvIds_.end());

    // via set TODO unordered
    std::set<UvType> uvNewTmp;
    for(size_t t = 0; t < threadpool_.nThreads(); ++t) {
        const auto & thisVector = threadVectors[t];
        for(const auto & elem : thisVector) {
            uvNewTmp.insert(elem.first);
        }
    }
    newUvIds_.resize(uvNewTmp.size());
    std::copy(uvNewTmp.begin(), uvNewTmp.end(), newUvIds_.begin());


    // construct a lut for the new uv-ids
    std::map<UvType, EdgeType> lut;
    for(EdgeType newEdgeId = 0; newEdgeId < newUvIds_.size(); ++newEdgeId) {
        lut[newUvIds_[newEdgeId]] = newEdgeId;
    }

    // find old to new edges in parallel by going over the unique new uv-ids
    parallel::parallel_foreach(threadpool_, threadpool_.nThreads(), [&](const int tId, const int t){

        const auto & thisVec = threadVectors[t];
        EdgeType newEdgeId;
        for(const auto & elem: thisVec) {
            newEdgeId = lut[elem.first];
            // NOTE: this is thread safe, because every old edge id (== elem.second) only occurs once
            oldToNewEdges_[elem.second] = newEdgeId;
        }

    });
    */

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
        const int64_t newEdge = oldToNewEdges_[edgeId];
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
