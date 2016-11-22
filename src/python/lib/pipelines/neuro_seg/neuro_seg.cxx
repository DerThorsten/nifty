
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include <iostream>

#include "nifty/pipelines/neuro_seg.hxx"
#include "nifty/tools/blocking.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace pipelines{
namespace neuro_seg{

    void exporBlockData( py::module & mod){

        py::class_<BlockData>(mod,"BlockData")

            .def("__init__",
                [](
                    BlockData & instance,
                    typename BlockData::BlockingType & blocking,
                    const size_t blockIndex
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        new (&instance) BlockData(blocking, blockIndex);
                    }
                }
            )
            .def("accumulate",[]
                (
                    BlockData & instance,
                    marray::PyView<uint32_t, 3> labels
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        instance.accumulate(labels);
                    }
                }
            )
            .def("merge",[]
                (
                    BlockData & instance,
                    const BlockData & other
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        instance.merge(other);
                    }
                }
            )
            ;
        ;
    }  

}
}
}




PYBIND11_PLUGIN(_neuro_seg) {
    py::module mod("_neuro_seg", "neuro seg submodule of nifty");

    using namespace nifty::pipelines::neuro_seg;


    exporBlockData(mod);
        
    return mod.ptr();
}

