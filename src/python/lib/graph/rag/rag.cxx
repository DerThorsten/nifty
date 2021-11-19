#include <pybind11/pybind11.h>
#include <iostream>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

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
    void exportAccumulateAffinityFeatures(py::module &);
    void exportComputeLiftedEdges(py::module &);
}
}


PYBIND11_MODULE(_rag, ragModule) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();
    ragModule.doc() = "rag submodule of nifty.graph";

    using namespace nifty::graph;
    exportGridRag(ragModule);
    exportGridRagStacked(ragModule);
    exportGraphAccumulator(ragModule);
    exportProjectToPixels(ragModule);
    exportAccumulate(ragModule);
    exportAccumulateStacked(ragModule);
    exportAccumulateFlat(ragModule);
    exportGridRagCoordinates(ragModule);
    exportAccumulateAffinityFeatures(ragModule);
    exportComputeLiftedEdges(ragModule);
    #ifdef WITH_FASTFILTERS
    exportAccumulateEdgeFeaturesFromFilters(ragModule);
    #endif
}
