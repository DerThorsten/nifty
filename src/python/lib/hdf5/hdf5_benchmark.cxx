#ifdef WITH_HDF5

#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/tools/blocking.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace py = pybind11;



namespace nifty{
namespace hdf5{


    void exportBenchmark(py::module & hdf5Module) {


        hdf5Module.def("runBenchmark",
        [](
            const nifty::hdf5::Hdf5Array<uint32_t> & data,
            const nifty::tools::Blocking<3> & blocking,
            const int numberOfThreads
        ){

            std::mutex lock;
            uint64_t val = 0;

            const auto numberOfBlocks = blocking.numberOfBlocks();
            nifty::parallel::parallel_foreach(
                numberOfThreads,
                numberOfBlocks,
                [&](
                    const int tid,
                    const int blockIndex
                ){
                    typedef typename xt::xarray<uint32_t>::shape_type ShapeType;
                    const auto block = blocking.getBlock(blockIndex);
                    const ShapeType blockShape(block.shape().begin(), block.shape().end());
                    lock.lock();
                    xt::xarray<uint32_t> subarray(blockShape);
                    data.readSubarray(block.begin().begin(), subarray);

                    const auto _val = subarray(0,0,0);

                    val += _val;
                    lock.unlock();
                }
            );
            // to make sure the above code is not optimized away
            std::cout<<"val "<<val<<"\n";
        }
        );

    }

}
}

#endif
