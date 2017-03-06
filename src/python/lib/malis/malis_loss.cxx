#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/malis/malis.hxx"

namespace py = pybind11;


namespace nifty {
namespace malis {

    template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
    void exportMalisLossT(py::module & malisModule){

        malisModule.def("malis_gradient",
           [](
                nifty::marray::PyView<DATA_TYPE, DIM> affinities,
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

    void exportMalisLoss(py::module & malisModule){
        exportMalisLossT<2,float,uint32_t>(malisModule);
        exportMalisLossT<3,float,uint32_t>(malisModule);
    }
}
}
