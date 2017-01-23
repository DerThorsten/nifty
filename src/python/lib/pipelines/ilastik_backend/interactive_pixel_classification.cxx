#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx>
#include <nifty/pipelines/ilastik_backend/input_type_tags.hxx>

namespace py = pybind11;

namespace nifty{
namespace pipelines{
namespace ilastik_backend{


    template<class INPUT_TYPE_TAG>
    void exportInteractivePixelClassificationT(py::module & module, const std::string & clsName) {
    
        typedef InteractivePixelClassification<INPUT_TYPE_TAG> IpcType;

        py::class_<IpcType>(module, clsName.c_str())
            .def("__init__",[](IpcType &instance) {
                new (&instance) IpcType();
            })
        ;
    }

    void exportInteractivePixelClassification(py::module & module) {
        exportInteractivePixelClassificationT<SpatialTag<2> >(module, "InteractivePixelClassificationSpatial2D");
        exportInteractivePixelClassificationT<SpatialTag<3> >(module, "InteractivePixelClassificationSpatial3D");
    }

}
}
}