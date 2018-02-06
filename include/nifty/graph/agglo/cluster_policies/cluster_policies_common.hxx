#pragma once

namespace nifty{
namespace graph{
namespace agglo{






struct EdgeWeightedClusterPolicySettings{
    double sizeRegularizer{0.5};
    uint64_t numberOfNodesStop{1};
    uint64_t numberOfEdgesStop{0};
};


inline bool isNegativeInf(const double val){
    return val < 0 && std::isinf(val);
}




} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

