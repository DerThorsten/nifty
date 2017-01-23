#ifdef WITH_HDF5


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <nifty/marray/marray.hxx>
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
        typedef InputDataBase<T, DIM, MULTICHANNEL> BaseType;
        typedef typename BaseType::Coord Coord;
        /* Trampoline (need one for each virtual function) */
        void readData(
            const Coord & begin, const Coord & end, nifty::marray::View<T> & out
        ) override {
            PYBIND11_OVERLOAD_PURE(
                void, /* Return type */
                BaseType,      /* Parent class */
                readData,          /* Name of function in C++ (must match Python name) */
                begin,end,out
            );
        }

    private:
    };

    template<class T, size_t DIM, bool MULTICHANNEL>
    void exportInputT(
        py::module & module,
        const std::string & namePrefix
    ){  
        typedef InputDataBase<T, DIM, MULTICHANNEL>     BaseType;
        typedef PyInputDataBase<T, DIM, MULTICHANNEL> PyBaseType;


        const auto inputBaseClsName = std::string("InputDataBase") + namePrefix;
        py::class_<BaseType, PyBaseType> inputBase(module, inputBaseClsName.c_str());
        
        inputBase
           .def(py::init<>())
        ;


        
        // export the hdf5 class WITH UINT8 as internal type
        typedef Hdf5Input<T, DIM, MULTICHANNEL, uint8_t> Hdf5InputType;
        const auto hdf5ClsName = std::string("Hdf5InputUInt32") + namePrefix;
        const auto hdf5FacName = std::string("hdf5InputUInt32") + namePrefix;
        py::class_<Hdf5InputType > hdf5Input(module, hdf5ClsName.c_str(),inputBase);

        hdf5Input
            .def("__init__",[](
                    Hdf5InputType &instance,
                    const nifty::hdf5::Hdf5Array<uint8_t> & data
            ) {
                new (&instance) Hdf5InputType(data);
            },
                py::keep_alive<0, 1>()
            )
        ;

        module.def(hdf5FacName.c_str(), [](
            const nifty::hdf5::Hdf5Array<uint8_t> & data
        ) {
            BaseType * ptr =  new Hdf5InputType(data);
            return ptr;
        },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>()
        );

        
    }



    template<class INPUT_TYPE_TAG, bool MULTICHANNEL>
    void exportInteractivePixelClassificationT(py::module & module, const std::string & clsName) {
    
        typedef InteractivePixelClassification<INPUT_TYPE_TAG, MULTICHANNEL> IpcType;
        py::class_<IpcType>(module, clsName.c_str())
            .def("__init__",[](IpcType &instance) {
                new (&instance) IpcType();
            })

            .def("addTrainingInstance",&IpcType::addTrainingInstance)
        ;
    }

    void exportInteractivePixelClassification(py::module & module) {

        // export the input datasets
        exportInputT<float,2,false>(module, "Float2D");
        exportInputT<float,3,false>(module, "Float3D");

        // export the class itself
        exportInteractivePixelClassificationT<SpatialTag<2>,false>(module, "InteractivePixelClassificationSpatial2D");
        exportInteractivePixelClassificationT<SpatialTag<3>,false>(module, "InteractivePixelClassificationSpatial3D");
    }

}
}
}

#endif