#pragma once
#ifndef NIFTY_PYTHON_CONVERTER_HXX
#define NIFTY_PYTHON_CONVERTER_HXX

#include <initializer_list>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nifty/marray/marray.hxx>

namespace py = pybind11;


namespace nifty{

    template<class T>
    std::vector<size_t> toSizeT(std::initializer_list<T> l){
        return std::vector<size_t>(l.begin(),l.end());
    }



    template<class T>
    class NumpyArray : public marray::View<T>{
    public:

        template<class SHAPE_T,class STRIDE_T>
        NumpyArray(
            std::initializer_list<SHAPE_T> shape,
            std::initializer_list<STRIDE_T> strides
        )
        : array_(){


            auto svec = toSizeT(strides);
            for(auto i=0; i<svec.size(); ++i){
                svec[i] *= sizeof(T);
            }

            array_  = py::array(
                py::buffer_info(NULL, sizeof(T),
                py::format_descriptor<T>::value,
                shape.size(), toSizeT(shape),svec)
            );

            py::buffer_info info = array_.request();
            T * ptr = static_cast<T*>(info.ptr);

            this->assign(shape.begin(),shape.end(),strides.begin(),ptr,
                marray::FirstMajorOrder
            );
        }


        NumpyArray(py::array_t<T> & array)
        :   array_(array) 
        {
            py::buffer_info info = array.request();
            T * ptr = static_cast<T*>(info.ptr);
            const auto & shape = info.shape;
            auto  strides=  info.strides;
            for(auto & s : strides)
                s/=sizeof(T);
            this->assign(shape.begin(),shape.end(),strides.begin(),ptr,
                marray::FirstMajorOrder
            );
        }
        

        py::array_t<T>  pyArray(){
            return array_;
        }        

    private:
        py::array_t<T> array_;
    };








} // namespace nifty

#endif  // NIFTY_PYTHON_CONVERTER_HXX
