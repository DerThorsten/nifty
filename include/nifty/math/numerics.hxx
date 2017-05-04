#pragma once

#include <limits>   // std::numeric_limits
#include "vigra/numerictraits.hxx"

namespace nifty{
namespace math{


    template<class T0, class T1>
    struct PromoteTraits{
        typedef typename vigra::PromoteTraits<T0, T1>::Promote          PromoteType; 
        typedef typename vigra::NumericTraits<PromoteType>::RealPromote RealPromoteType; 
    };

    template<class T>
    struct NumericTraits : public vigra::NumericTraits<T>{
        typedef vigra::NumericTraits<T> BaseType;
        typedef typename BaseType::Promote PromoteType;
        typedef typename BaseType::RealPromote RealPromoteType;
    };



    template<class T, bool IS_NUMBER>
    class NumericsImplDispatch;

    template<class T>
    class NumericsImplDispatch<T, true>{
    public:

        typedef typename vigra::NumericTraits<T>::RealPromote RealPromoteType; 

        static constexpr T zero(){
            return  static_cast<T>(0);
        }
        static constexpr T one(){
            return  static_cast<T>(1);
        }
        static void zero(T & value){
            value = static_cast<T>(0);
        }
        static T one(T & value){
            value =  static_cast<T>(1);
        }


        static RealPromoteType real(const T & value){
            return static_cast<RealPromoteType>(value);
        }

    };


    template<class T>
    class Numerics : 
        public NumericsImplDispatch<
            T, std::numeric_limits<T>::is_specialized
    >{
    };  



    // explicit specialization
    



}
}