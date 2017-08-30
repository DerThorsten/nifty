// sleep without gil
//
#include <chrono>
#include <thread>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace tools{

void exportSleep(py::module & toolsModule) {
    toolsModule.def("sleepMilliseconds",[](size_t nMilliseconds) {
        py::gil_scoped_release allowThreads;
        std::this_thread::sleep_for(std::chrono::milliseconds(nMilliseconds));
    },
    py::arg("nMilliseconds")
    );
}

}
}
