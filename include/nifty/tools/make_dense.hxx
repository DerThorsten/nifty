#pragma once
#include <unordered_map>


#include "nifty/xtensor/xtensor.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace tools{

    template<class ARRAY>
    void makeDense(xt::xexpression<ARRAY> & dataExp){
        typedef typename ARRAY::value_type DataType;
        auto & data = dataExp.derived_cast();

        std::unordered_map<DataType, DataType> hmap;
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            hmap[val] = DataType();
        }
        DataType c = 0;
        for(auto & kv : hmap){
            kv.second = c;
            ++c;
        }
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            data(s) = hmap[val];
        }
    }

    template<class ARRAY1, class ARRAY2>
    void makeDense(
        const xt::xexpression<ARRAY1> & dataInExp,
        xt::xexpression<ARRAY2> & dataOutExp
    ){
        const auto & dataIn = dataInExp.derived_cast();
        auto & dataOut = dataOutExp.derived_cast();
        for(auto s=0; s<dataIn.size(); ++s){
            dataOut(s) = dataIn(s);
        }
        makeDense(dataOut);
    }

}
}

