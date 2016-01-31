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


    // Addition
    template< typename T, typename Op1 , typename Op2 >
    class VecAdd {
    private:
       const Op1 &op1;
       const Op2 &op2;

    public:
       VecAdd(const Op1 &a, const Op2 &b ) :
          op1(a), op2(b) {}

       T operator[](const size_t i) const
       { return op1[i] + op2[i]; }

       size_t size() const
       { return op1.size(); }
    };

  


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
    } 

    NIFTY_MACRO_BINARY_OP_INPLACE(+=);
    NIFTY_MACRO_BINARY_OP_INPLACE(-=);
    NIFTY_MACRO_BINARY_OP_INPLACE(*=);
    NIFTY_MACRO_BINARY_OP_INPLACE(/=);

    #undef NIFTY_MACRO_BINARY_OP_INPLACE





    template<class T, unsigned int DIM>
    class StaticArray : public std::array<T,DIM>{
    public:
        //using std::array<T,DIM>::array;
        StaticArray()
        :   std::array<T,DIM>(){}

        template <typename... Args>
        StaticArray(Args &&... args) : std::array<T,DIM>({std::forward<Args>(args)...}) {
        }
    };






} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX
