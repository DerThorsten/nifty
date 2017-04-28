#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGridRag(py::module &);
    void exportGridRagStacked(py::module &);
    void exportGraphAccumulator(py::module &);
    void exportProjectToPixels(py::module &);
    void exportAccumulate(py::module &);
    void exportAccumulateStacked(py::module &);
    void exportAccumulateEdgeFeaturesFromFilters(py::module &);
    void exportAccumulateFlat(py::module &);
    void exportGridRagCoordinates(py::module &);

}
}


PYBIND11_PLUGIN(_rag) {
    py::module ragModule("_rag", "rag submodule of nifty.graph");

    using namespace nifty::graph;

    exportGridRag(ragModule);
    exportGridRagStacked(ragModule);
    exportGraphAccumulator(ragModule);
    exportProjectToPixels(ragModule);
    exportAccumulate(ragModule);
    exportAccumulateStacked(ragModule);
    exportAccumulateFlat(ragModule);
    exportGridRagCoordinates(ragModule);
    #ifdef WITH_FASTFILTERS
    exportAccumulateEdgeFeaturesFromFilters(ragModule);
    #endif

    return ragModule.ptr();
}

