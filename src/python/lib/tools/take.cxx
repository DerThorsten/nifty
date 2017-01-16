#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <typeinfo> // to debug atm

#include "nifty/python/converter.hxx"
#include "nifty/tools/make_dense.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T, bool AUTO_CAST>
    void exportTakeT(py::module & toolsModule) {

        toolsModule.def("_take",
        [](
           nifty::marray::PyView<T,1,AUTO_CAST> relabeling,
           nifty::marray::PyView<T,1,AUTO_CAST> toRelabel
        ){

            nifty::marray::PyView<T> out(toRelabel.shapeBegin(), toRelabel.shapeEnd());
            {
                py::gil_scoped_release allowThreads;
                for(size_t i=0; i<toRelabel.shape(0); ++i){
                    out[i] = relabeling[toRelabel[i]];
                }
                
            }
            return out;
        });
    }


    void exportTake(py::module & toolsModule) {
        
        exportTakeT<uint32_t, false>(toolsModule);
        exportTakeT<uint64_t, false>(toolsModule);
        exportTakeT<int32_t, false>(toolsModule);

        //exportTakeT<float   , false>(toolsModule);

        exportTakeT<int64_t   , true>(toolsModule);

    }

}
}
