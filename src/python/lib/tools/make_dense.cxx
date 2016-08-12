#ifdef WITH_HDF5
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
    void exportMakeDenseT(py::module & toolsModule) {

        toolsModule.def("makeDense",
        [](
           nifty::marray::PyView<T,0,AUTO_CAST> dataIn
        ){
            std::cout<<"typeinfo "<<typeid(T).name()<<"\n";
            nifty::marray::PyView<T> dataOut(dataIn.shapeBegin(), dataIn.shapeEnd());
            tools::makeDense(dataIn, dataOut);
            return dataOut;
        });
    }


    void exportMakeDense(py::module & toolsModule) {
        
        exportMakeDenseT<uint32_t, false>(toolsModule);
        exportMakeDenseT<uint64_t, false>(toolsModule);
        exportMakeDenseT<int32_t, false>(toolsModule);

        //exportMakeDenseT<float   , false>(toolsModule);

        exportMakeDenseT<int64_t   , true>(toolsModule);

    }

}
}

#endif
