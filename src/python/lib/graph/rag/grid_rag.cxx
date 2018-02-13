#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

// still need this for python bindings of nifty::ArrayExtender
#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    template<class CLS, class BASE>
    void removeFunctions(py::class_<CLS, BASE > & clsT){
        clsT
            .def("insertEdge", [](CLS * self,const uint64_t u,const uint64_t ){
                throw std::runtime_error("cannot insert edges into 'GridRag'");
            })
            .def("insertEdges",[](CLS * self, py::array_t<uint64_t> pyArray) {
                throw std::runtime_error("cannot insert edges into 'GridRag'");
            })
        ;
    }


    template<size_t DIM, class LABELS>
    void exportGridRagT(py::module & ragModule,
                        const std::string & clsName,
                        const std::string & facName){
        // typedefs
        typedef UndirectedGraph<> BaseGraph;
        typedef LABELS LabelsType;
        typedef GridRag<DIM, LABELS> GridRagType;

        // export the rag
        auto clsT = py::class_<GridRagType, BaseGraph>(ragModule, clsName.c_str());
        clsT
            .def_property_readonly("shape",[](const GridRagType & self){return self.shape();})
        ;
        removeFunctions<GridRagType, BaseGraph>(clsT);

        // factories
        // from labels
        ragModule.def(facName.c_str(),[](const LabelsType & labels,
                                         const int64_t numberOfLabels,
                                         const std::array<int64_t, DIM> blockShape,
                                         const int numberOfThreads){

                auto s = typename GridRagType::SettingsType();
                for(int ii = 0; ii < DIM; ++ii) {
                    s.blockShape[ii] = blockShape[ii];
                }

                s.numberOfThreads = numberOfThreads;
                return new GridRagType(labels, numberOfLabels, s);
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels").noconvert(),
            py::arg("numberOfLabels"),
            py::arg("blockShape"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

        // from labels + serialization
        ragModule.def(facName.c_str(),[](const LabelsType & labels,
                                         const int64_t numberOfLabels,
                                         const xt::pytensor<uint64_t, 1> & serialization){

                auto  startPtr = &serialization(0);
                auto  lastElement = &serialization(serialization.size()-1);
                auto d = lastElement - startPtr + 1;

                NIFTY_CHECK_OP(d, ==, serialization.size(), "serialization must be contiguous");

                auto s = typename  GridRagType::SettingsType();
                s.numberOfThreads = -1;
                return new GridRagType(labels, numberOfLabels, startPtr, s);
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels").noconvert(),
            py::arg("numberOfLabels"),
            py::arg("serialization")
        );

    }


    void exportGridRag(py::module & ragModule) {

        // export grid rag with in-memory labels
        typedef xt::pytensor<uint32_t, 2> ExplicitPyLabels2D;
        exportGridRagT<2, ExplicitPyLabels2D>(ragModule,
                                              "ExplicitLabelsGridRag2D",
                                              "explicitLabelsGridRag2D");

        typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D32;
        exportGridRagT<3, ExplicitPyLabels3D32>(ragModule,
                                               "ExplicitLabelsGridRag3D32",
                                               "explicitLabelsGridRag3D32");

        typedef xt::pytensor<uint64_t, 3> ExplicitPyLabels3D64;
        exportGridRagT<3, ExplicitPyLabels3D64>(ragModule,
                                                "ExplicitLabelsGridRag3D64",
                                                "explicitLabelsGridRag3D64");

        // FIXME need hdf5 with xtensor support for this to work
        // export grid rag with hdf5 labels
        //#ifdef WITH_HDF5
        //typedef nifty::hdf5::Hdf5Array<uint32_t> Hdf5Labels2D;
        //exportGridRagT<2, Hdf5Labels2D>(ragModule,
        //                                "GridRag2DHdf5",
        //                                "gridRag2DHdf5");

        //typedef nifty::hdf5::Hdf5Array<uint32_t> Hdf5Labels3D32;
        //exportGridRagT<3, Hdf5Labels3D32>(ragModule,
        //                                  "GridRag3DHdf532",
        //                                  "gridRag3DHdf532");

        //typedef nifty::hdf5::Hdf5Array<uint64_t> Hdf5Labels3D64;
        //exportGridRagT<3, Hdf5Labels3D64>(ragModule,
        //                                  "GridRag3DHdf564",
        //                                  "gridRag3DHdf564");
        //#endif

        // export with z5 labels
        #ifdef WITH_Z5
        typedef nifty::nz5::DatasetWrapper<uint32_t> Z5Labels2D;
        exportGridRagT<2, Z5Labels2D>(ragModule,
                                      "GridRag2DZ5",
                                      "gridRag2DZ5");
        //
        typedef nifty::nz5::DatasetWrapper<uint32_t> Z5Labels3D32;
        exportGridRagT<3, Z5Labels3D32>(ragModule,
                                        "GridRag3DZ532",
                                        "gridRag3DZ532");

        typedef nifty::nz5::DatasetWrapper<uint64_t> Z5Labels3D64;
        exportGridRagT<3, Z5Labels3D64>(ragModule,
                                        "GridRag3DZ564",
                                        "gridRag3DZ564");
        #endif
    }

} // end namespace graph
} // end namespace nifty
