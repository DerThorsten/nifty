#pragma once

#include <pybind11/pybind11.h>

namespace nifty{
namespace cgp{

template<
    class CLS
>
inline  void exportCellVector(pybind11::module & m, pybind11::class_<CLS> & pyCls) {
    pyCls
        .def("__getitem__", [](const CLS & self, uint32_t i){
            return self[i];
        },
            pybind11::return_value_policy::reference_internal
        )
    ;
}

}
}