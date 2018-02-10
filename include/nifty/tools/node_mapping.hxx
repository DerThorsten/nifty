#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "nifty/xtensor/xtensor.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty {
namespace tools {

    template<class NODE_ARRAY>
    inline void mapNodeLabeling(const xt::xexpression<NODE_ARRAY> & nodeLabelingExp
                                const xt::xexpression<NODE_ARRAY> & initialNodeLabelingExp
                                xt::xexpression<NODE_ARRAY> & newInitialNodeLabelingExp,
                                const int numberOfThreads=-1) {
        //
        typedef typename NODE_ARRAY::value_type NodeType;
        const auto & nodeLabeling = nodeLabelingExp.derived_cast();
        const auto & initialNodeLabeling = initialNodeLabelingExp.derived_cast();
        auto & newInitialNodeLabeling = newInitialNodeLabelingExp.derived_cast();

        nifty::parallel::threadpool::ThreadPool threadpool(numberOfThreads);
        const size_t nThreads = threadpool.nThreads();
        const size_t nInitialNodes = initialNodeLabeling.shape()[0]

        nifty::parallel::parallel_foreach(threadpool, nInitialNodes, [&](const int tId, const NodeType initialNode){
            const NodeType oldNodeLabel = initialNodeLabeling(initialNode);
            const NodeType newNodeLabel = nodeLabeling(oldNodeLabel);
            newInitialNodeLabeling(initialNode) = newNodeLabel;
        });

        return newInitialNodeLabeling;
    }


}
}
