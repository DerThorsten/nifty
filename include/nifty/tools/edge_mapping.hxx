#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "nifty/marray/marray.hxx"

namespace nifty {
namespace tools {

template<class EDGE_TYPE, class NODE_TYPE>
class EdgeMapping {

public:

    typedef EDGE_TYPE EdgeType;
    typedef NODE_TYPE NodeType;
    typedef std::pair<NodeType, NodeType> UvType;
    typedef std::vector<UvType> UvVectorType;

    EdgeMapping(const size_t numberOfEdges)
        : oldToNewEdges_(numberOfEdges), newUvIds_()
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
}


// TODO this can be parallelized
// TODO check edge values for proper length
template<class EDGE_TYPE, class NODE_TYPE>
template<class T>
void EdgeMapping<EDGE_TYPE, NODE_TYPE>::mapEdgeValues(
        const std::vector<T> & edgeValues, std::vector<T> & newEdgeValues) const {

    newEdgeValues.clear();
    newEdgeValues.resize(newUvIds_.size(), 0);

    EdgeType newEdge;
    for(size_t i = 0; i < edgeValues.size();  ++i) {
        newEdge = oldToNewEdges_[i];
        if(newEdge == -1) {
            continue;
        }
        newEdgeValues[newEdge] += edgeValues[i];
    }
}


// TODO this can be easily parallelized
template<class EDGE_TYPE, class NODE_TYPE>
void  EdgeMapping<EDGE_TYPE, NODE_TYPE>::getNewEdgeIds(
        const std::vector<EdgeType> & edgeIds, std::vector<EdgeType> & newEdgeIds) const {
    
    newEdgeIds.clear();
    newEdgeIds.reserve(edgeIds.size());
    EdgeType newEdge;
    for(auto edgeId : edgeIds) {
        newEdge = oldToNewEdges_[edgeId];
        if(newEdge != -1) {
            newEdgeIds.push_back(newEdge);
        }
    }

    std::sort(newEdgeIds.begin(), newEdgeIds.end());
    auto last = std::unique(newEdgeIds.begin(), newEdgeIds.end());
    newEdgeIds.erase(last, newEdgeIds.end());
}


}
}
