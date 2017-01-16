#pragma once

#ifdef WITHIN_TRAVIS
#include <unordered_map>
#define __nifty_mapimpl__ std::unordered_map
#else
#include <boost/container/flat_map.hpp>
#define __nifty_mapimpl__ boost::container::flat_map
#endif

namespace nifty {
namespace container{

    template<class KEY, class VALUE>
    using BoostFlatMap = __nifty_mapimpl__<KEY, VALUE>;

} // container
} // namespace nifty

