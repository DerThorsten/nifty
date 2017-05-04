#pragma once

#include <array>
#include <vector>



#include "nifty/math/numerics.hxx"
#include "nifty/array/static_array.hxx"















namespace nifty{
namespace math{
    
    template<class T0, class T1, size_t N>
    typename nifty::math::PromoteTraits<T0,T1>::RealPromoteType 
    euclideanDistance(
        const nifty::array::StaticArray<T0, N> & a,
        const nifty::array::StaticArray<T1, N> & b
    ){
        typedef typename nifty::math::PromoteTraits<T0,T1>::PromoteType  T0T1;

        auto d = Numerics<T0T1>::zero();

        for(size_t i=0; i<N; ++i){
            
            const auto dist = T0T1(a[i])-T0T1(b[i]);
            d += dist*dist;
        }
        return std::sqrt(d);
    }


}
}