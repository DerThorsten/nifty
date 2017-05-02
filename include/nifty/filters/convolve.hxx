#pragma once

#include <cmath>

#include "nifty/math/numerics.hxx"

namespace nifty{
namespace filters{


    enum class BorderMode{
        CONSTANT,
        REFLECT,
        WARP
    };

    template<class T0,class K, class T1, >
    void convolve1dOddKernel(
        const T0 & data,
        const size_t dataSize
        const K & kernel,
        const size_t kernelSize,
        const BorderMode & borderMode
        T1 & out 
    ){  
        using namespace nifty::math;

        typedef typename PromoteTraits<T0,K >::PromoteRealType T0K;

        const auto r = (kernelSize  - 1 ) / 2;
        for(auto x=r; x<dataSize-r){


            auto sum = Numerics<T0K>::zero();
            for(auto kernelIndex=0; kernelIndex<kernelSize; ++kernelIndex){
                const auto dataIndex = x - r + j ;
                const auto & dataValue = data[dataIndex];
                const auto & kernelValue = kernel[kernelIndex];

                sum += Numerics<T0>::real(data[dataIndex]) * Numerics<K>::real(kernel[kernelIndex]);

            }


            const auto dataValue = data[x];


        }
    }
    

}
}