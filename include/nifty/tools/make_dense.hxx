#pragma once
#include <unordered_map>


#include "nifty/marray/marray.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace tools{

    template<class T>
    void makeDense(
        marray::View<T> & data
    ){
        std::unordered_map<T,T> hmap;
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            hmap[val] = T();
        }
        T c=0;
        for(auto & kv : hmap){
            kv.second = c;
            ++c;
        }
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            data(s) = hmap[val];
        }
    }

    template<class T>
    void makeDense(
        const marray::View<T> & dataIn,
        marray::View<T> & dataOut
    ){
        for(auto s=0; s<dataIn.size(); ++s){
            dataOut(s) = dataIn(s);
        }
        makeDense(dataOut);
    }

}
}

