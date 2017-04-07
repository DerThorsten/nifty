#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "nifty/deep_learning/malis.hxx"
#include "nifty/python/converter.hxx"


namespace nifty{
namespace deep_learning{

    template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
    void exportMalisGradientT(py::module & module){

        module.def("malis_gradient",
           [](
                nifty::marray::PyView<DATA_TYPE, DIM+1> affinities,
                nifty::marray::PyView<LABEL_TYPE, DIM> groundtruth
           ){  
                typedef nifty::array::StaticArray<int64_t,DIM+1> Coord;
                Coord shape;
                for(int d = 0; d < DIM+1; ++d)
                    shape[d] = affinities.shape(d);
                nifty::marray::PyView<size_t, DIM+1> positiveGradients(shape.begin(), shape.end());
                nifty::marray::PyView<size_t, DIM+1> negativeGradients(shape.begin(), shape.end());
                {
                    py::gil_scoped_release allowThreads;
                    compute_malis_gradient<DIM>(affinities, groundtruth, positiveGradients, negativeGradients);
                }
                return std::make_tuple(positiveGradients, negativeGradients);
           },
           py::arg("affinities"),
           py::arg("groundtruth")
        );
    }

    void exportMalis(py::module & module) {
        exportMalisGradientT<2,float,uint32_t>(module);
        exportMalisGradientT<3,float,uint32_t>(module);
    }

}
}
