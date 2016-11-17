#pragma once
#ifndef NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX
#define NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX

#include <array>
#include <vector>

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
        return a; \
    } \
    template<class ARRAY_CLASS> \
    ArrayExtender<ARRAY_CLASS> & operator operatorSymbol ( \
        ArrayExtender<ARRAY_CLASS> & a, \
        typename ArrayExtender<ARRAY_CLASS>::const_reference  b \
    ){ \
        for(auto i=0; i<a.size(); ++i){ \
            a[i] operatorSymbol b; \
        } \
        return a; \
    } 

    NIFTY_MACRO_BINARY_OP_INPLACE(+=);
    NIFTY_MACRO_BINARY_OP_INPLACE(-=);
    NIFTY_MACRO_BINARY_OP_INPLACE(*=);
    NIFTY_MACRO_BINARY_OP_INPLACE(/=);
    NIFTY_MACRO_BINARY_OP_INPLACE(&=);
    NIFTY_MACRO_BINARY_OP_INPLACE(|=);
    #undef NIFTY_MACRO_BINARY_OP_INPLACE


    #define NIFTY_MACRO_BINARY_OP(operatorSymbol, inplaceSymbol) \
    template<class ARRAY_CLASS> \
    ArrayExtender<ARRAY_CLASS>  operator operatorSymbol ( \
        const ArrayExtender<ARRAY_CLASS> & a, \
        const ArrayExtender<ARRAY_CLASS> & b \
    ){ \
        NIFTY_ASSERT_OP( a.size(),==,b.size()); \
        auto res = a; \
        res inplaceSymbol b; \
        return res; \
    } \
    template<class ARRAY_CLASS> \
    ArrayExtender<ARRAY_CLASS>  operator operatorSymbol ( \
        const ArrayExtender<ARRAY_CLASS> & a, \
        const typename ArrayExtender<ARRAY_CLASS>::const_reference & b \
    ){ \
        auto res = a; \
        for(auto i=0; i<a.size(); ++i){ \
            res[i] inplaceSymbol b; \
        } \
        return res; \
    } \

    NIFTY_MACRO_BINARY_OP(+, +=);
    NIFTY_MACRO_BINARY_OP(-, -=);
    NIFTY_MACRO_BINARY_OP(*, *=);
    NIFTY_MACRO_BINARY_OP(/, /=);
    NIFTY_MACRO_BINARY_OP(&, &=);
    NIFTY_MACRO_BINARY_OP(|, |=);
    #undef NIFTY_MACRO_BINARY_OP



    // to give std::array a proper constructor
    // since it is an aggregate we need
    // to impl. this 
    // => we are giving up the aggregate status
    template<class T, size_t DIM>
    class StaticArrayBase : public std::array<T,DIM>{
    public:
        typedef  std::array<T,DIM> BaseType;
        //using std::array<T,DIM>::array;
        StaticArrayBase()
        :   std::array<T,DIM>(){}

        StaticArrayBase(const T & value)
        :   std::array<T,DIM>(){
           std::fill(this->begin(), this->end(), value);
        }

        template<class INIT_T>
        StaticArrayBase(const std::initializer_list<INIT_T> & list){
            std::copy(list.begin(), list.end(), this->begin());
        }


        //template <typename... Args>
        //StaticArrayBase(Args &&... args) : std::array<T,DIM>({std::forward<Args>(args)...}) {
        //}

        const BaseType & asStdArray()const{
            return  static_cast< const BaseType & >(*this);
        }
        BaseType & asStdArray(){
            return  static_cast< BaseType & >(*this);
        }
    };


    template<class T,size_t SIZE>
    using StaticArray = ArrayExtender< StaticArrayBase<T,SIZE> >; 

    template<class T, class ALLOCATOR = std::allocator<T> >
    using Vector = ArrayExtender< std::vector<T,ALLOCATOR> >; 


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_ARRAY_ARITHMETIC_ARRAY_HXX
