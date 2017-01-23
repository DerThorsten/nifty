#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx>
#include <nifty/pipelines/ilastik_backend/input_type_tags.hxx>

namespace py = pybind11;

namespace nifty{
namespace pipelines{
namespace ilastik_backend{


    template<class T, size_t DIM, bool MULTICHANNEL>
    class PyInputDataBase
    : public InputDataBase<T, DIM, MULTICHANNEL>{
    public:
        //using PyInputDataBase::PyInputDataBase;

    private:
    };

    template<class T, size_t DIM, bool MULTICHANNEL>
    void exportInputT(
        py::module & module,
        const std::string & namePrefix
    ){  
        typedef InputDataBase<T, DIM, MULTICHANNEL>     BaseType;
        typedef PyInputDataBase<T, DIM, MULTICHANNEL> PyBaseType;


        // export the hdf5 class
        const auto hdf5ClsName = std::string("Hdf5Input") + namePrefix;
        py::class_<BaseType, PyBaseType> hdf5Input(module, hdf5ClsName.c_str());

        //hdf5Input
        //    .def("__init__",[])

    }



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

        // export the input datasets
        exportInputT<float,2,false>(module, "Float2D");
        exportInputT<float,3,false>(module, "Float3D");

        // export the class itself
        exportInteractivePixelClassificationT<SpatialTag<2> >(module, "InteractivePixelClassificationSpatial2D");
        exportInteractivePixelClassificationT<SpatialTag<3> >(module, "InteractivePixelClassificationSpatial3D");
    }

}
}
}