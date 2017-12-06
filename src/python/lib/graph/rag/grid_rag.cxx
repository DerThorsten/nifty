#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
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


    template<size_t DIM, class LABELS_PROXY>
    void exportGridRagT(py::module & ragModule,
                        const std::string & clsName,
                        const std::string & facName){
        // typedefs
        typedef UndirectedGraph<> BaseGraph;
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LabelsProxyType::LabelArrayType LabelArrayType;
        typedef GridRag<DIM, LABELS_PROXY> GridRagType;

        // export the labels proxy
        const auto labelsProxyClsName = clsName + std::string("LabelsProxy");
        const auto labelsProxyFacName = facName + std::string("LabelsProxy");
        py::class_<LabelsProxyType>(ragModule, labelsProxyClsName.c_str())
            .def("labels", &LabelsProxyType::labels, py::return_value_policy::reference)
        ;

        ragModule.def(labelsProxyFacName.c_str(),[](const LabelArrayType & labels,
                                                    const int64_t numberOfLabels){
                auto ptr = new LabelsProxyType(labels, numberOfLabels);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("numberOfLabels")
        );

        // export the rag
        auto clsT = py::class_<GridRagType,BaseGraph>(ragModule, clsName.c_str());
        clsT
            .def_property_readonly("shape",[](const GridRagType & self){return self.shape();})
        ;
        removeFunctions<GridRagType, BaseGraph>(clsT);

        // factories
        // from labels
        ragModule.def(facName.c_str(),[](const LabelsProxyType & labels,
                                         const int numberOfThreads){
                auto s = typename  GridRagType::SettingsType();
                s.numberOfThreads = numberOfThreads;
                auto ptr = new GridRagType(labels, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

        // TODO switch to xtensor
        // from labels + serialization
        ragModule.def(facName.c_str(),[](const LabelsProxyType & labels,
                                         nifty::marray::PyView<uint64_t, 1, false> serialization){

                auto  startPtr = &serialization(0);
                auto  lastElement = &serialization(serialization.size()-1);
                auto d = lastElement - startPtr + 1;

                NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");

                auto s = typename  GridRagType::SettingsType();
                s.numberOfThreads = -1;
                auto ptr = new GridRagType(labels, startPtr, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("serialization")
        );

    }


    void exportGridRag(py::module & ragModule) {

        // export grid rag with in-memory labels
        typedef ExplicitLabels<2, uint32_t> ExplicitLabels2DType;
        exportGridRagT<2, ExplicitLabels2DType>(ragModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        typedef ExplicitLabels<3, uint32_t> ExplicitLabels3DType;
        exportGridRagT<3, ExplicitLabels3DType>(ragModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");

        // export grid rag with hdf5 labels
        #ifdef WITH_HDF5
        typedef Hdf5Labels<2, uint32_t> Hdf5Labels2DType;
        exportGridRagT<2, Hdf5Labels2DType>(ragModule, "GridRagHdf5Labels2D", "gridRag2DHdf5");
        typedef Hdf5Labels<3, uint32_t> Hdf5Labels3DType;
        exportGridRagT<3, Hdf5Labels3DType>(ragModule, "GridRagHdf5Labels3D", "gridRag3DHdf5");
        #endif

        // export with z5 labels
        //#ifdef WITH_Z5
        //#endif
    }

} // end namespace graph
} // end namespace nifty
