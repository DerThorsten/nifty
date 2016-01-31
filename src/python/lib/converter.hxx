#pragma once
#ifndef NIFTY_PYTHON_CONVERTER_HXX
#define NIFTY_PYTHON_CONVERTER_HXX

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nifty/marray/marray.hxx>
namespace py = pybind11;


namespace nifty{


    template<class T>
    class NumpyArray : public marray::View<T>{
    public:
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
    private:
        py::array_t<T> array_;
    };

} // namespace nifty

#endif  // NIFTY_PYTHON_CONVERTER_HXX
