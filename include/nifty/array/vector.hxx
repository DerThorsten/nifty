#pragma once

#include "arithmetic_array.hxx"


namespace nifty{
namespace array{

    template<class T, class ALLOCATOR = std::allocator<T> >
    using Vector = ArrayExtender< std::vector<T,ALLOCATOR> >; 

}
}