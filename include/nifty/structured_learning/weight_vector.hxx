#pragma once
#ifndef NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX
#define NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX

#include <vector>
#include <string>

namespace nifty{
namespace structured_learning{

    template<class T>
    class WeightVector{


    public:
        const T & operator[](const std::size_t i)const{
            return weightValues_[i];
        }
        T & operator[](const std::size_t i){
            return weightValues_[i];
        }

        const std::string & value(const std::size_t i){
            return weightNames_[i];
        }
        const std::string & name(const std::size_t i){
            return weightNames_[i];
        }   

        const size_t size()const{
            return weightValues_.size();
        }

    private:
        std::vector<T>           weightValues_;
        std::vector<std::string> weightNames_;
    }

}
}

#endif // NIFTY_STRUCTURED_LEARNING_INSTANCES_WEIGHTED_EDGE_HXX