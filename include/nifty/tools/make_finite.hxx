#pragma once

#include <algorithm> 



namespace nifty{
namespace pipelines{
namespace tools{


    template<class T,class U>
    inline T 
    makeFinite(const T & val, const U & replaceVal){
        if(std::isfinite(val))
            return val;
        else
            return static_cast<T>(replaceVal);
    }

}
}
}