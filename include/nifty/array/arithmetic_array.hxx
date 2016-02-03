#pragma once
#ifndef NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX
#define NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX

#include <array>
#include "nifty/tools/runtime_check.hxx"

namespace nifty{
namespace array{
    

    template<class ARRAY_CLASS>
    class ArrayExtender : public ARRAY_CLASS
    {
    public:
        using ARRAY_CLASS::ARRAY_CLASS;

    private:

    };

    template<class STREAM, class ARRAY_CLASS>
    STREAM& operator << (STREAM &out, const ArrayExtender<ARRAY_CLASS> & array){
        out<<"[";
        auto first = true;
        for(const auto & val : array){
            if (first){
                first = false;
                out<<" "<<val;
            }
            else
                out<<", "<<val;
        }
        out<<" ]";
        return out;
    }

  


    #define NIFTY_MACRO_BINARY_OP_INPLACE(operatorSymbol) \
    template<class ARRAY_CLASS> \
    ArrayExtender<ARRAY_CLASS> & operator operatorSymbol ( \
        ArrayExtender<ARRAY_CLASS> & a, \
        const ArrayExtender<ARRAY_CLASS> & b \
    ){ \
        NIFTY_ASSERT_OP( a.size(),==,b.size()); \
        for(auto i=0; i<a.size(); ++i){ \
            a[i] operatorSymbol b[i]; \
        } \
    } \
    template<class ARRAY_CLASS> \
    ArrayExtender<ARRAY_CLASS> & operator operatorSymbol ( \
        ArrayExtender<ARRAY_CLASS> & a, \
        typename ArrayExtender<ARRAY_CLASS>::const_reference  b \
    ){ \
        for(auto i=0; i<a.size(); ++i){ \
            a[i] operatorSymbol b; \
        } \
    } 

    NIFTY_MACRO_BINARY_OP_INPLACE(+=);
    NIFTY_MACRO_BINARY_OP_INPLACE(-=);
    NIFTY_MACRO_BINARY_OP_INPLACE(*=);
    NIFTY_MACRO_BINARY_OP_INPLACE(/=);
    NIFTY_MACRO_BINARY_OP_INPLACE(&=);
    NIFTY_MACRO_BINARY_OP_INPLACE(|=);
    #undef NIFTY_MACRO_BINARY_OP_INPLACE




    // to give std::array a proper constructor
    // since it is an aggregate we need
    // to impl. this 
    // => we are giving up the aggregate status
    template<class T, unsigned int DIM>
    class StaticArrayBase : public std::array<T,DIM>{
    public:
        //using std::array<T,DIM>::array;
        StaticArrayBase()
        :   std::array<T,DIM>(){}

        template <typename... Args>
        StaticArrayBase(Args &&... args) : std::array<T,DIM>({std::forward<Args>(args)...}) {
        }
    };


    template<class T,unsigned int SIZE>
    using StaticArray = ArrayExtender< StaticArrayBase<T,SIZE> >; 

    template<class T, class ALLOCATOR = std::allocator<T> >
    using Vector = ArrayExtender< std::vector<T,ALLOCATOR> >; 


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX
