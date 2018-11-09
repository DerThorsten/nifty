#pragma once

#include "arithmetic_array.hxx"


namespace nifty{
namespace array{

    template<class T,size_t SIZE>
    using StaticArray = ArrayExtender< StaticArrayBase<T,SIZE> >;


}
}
