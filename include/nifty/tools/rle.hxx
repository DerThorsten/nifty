#pragma once
#include <vector>

#include "nifty/xtensor/xtensor.hxx"


namespace nifty{
namespace tools{


    template<class ARRAY>
    void computeRLE(xt::xexpression<ARRAY> & dataExp, std::vector<int> & counts){
        typedef typename ARRAY::value_type DataType;
        auto & data = dataExp.derived_cast();

        auto val = data[0];
        if(val == 1) {
            counts.push_back(0);
        }

        int count = 0;
        for(const auto & m : data) {
            if(val == m) {
                ++count;
            } else {
                val = m;
                counts.push_back(count);
                count = 1;
            }
        }
        counts.push_back(count);

    }


}
}
