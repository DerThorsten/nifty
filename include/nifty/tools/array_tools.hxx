#pragma once

#include <algorithm>
#include <vector>

#include "nifty/marray/marray.hxx"

namespace nifty {
namespace tools {

    template<class T>
    void uniques(const marray::View<T> & array, std::vector<T> & out){
        
        out.resize(array.size());
        std::copy(array.begin(), array.end(), out.begin());
        
        std::sort(out.begin(),out.end());
        auto last = std::unique(out.begin(), out.end());
        out.erase( last, out.end() );
    }


} // namespace tools
} // namespace nifty
