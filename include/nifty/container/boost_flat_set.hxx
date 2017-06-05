#pragma once



#ifdef WITHIN_TRAVIS
#include "nifty/container/flat_set.hxx"
#define __nifty_setimpl__ nifty::container::FlatSet
#else
#include <boost/container/flat_set.hpp>
#define __nifty_setimpl__ boost::container::flat_set
#endif

namespace nifty {
namespace container{

    template<class T>
    using BoostFlatSet = __nifty_setimpl__<T>;

} // container
} // namespace nifty
  
