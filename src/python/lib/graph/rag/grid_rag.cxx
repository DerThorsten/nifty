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

     

    template<size_t DIM, class LABELS>
    void exportExpilictGridRagT(
        py::module & ragModule, 
        const std::string & clsName,
        const std::string & facName
    ){
        typedef UndirectedGraph<> BaseGraph;
        typedef ExplicitLabelsGridRag<DIM, LABELS> GridRagType;

        auto clsT = py::class_<GridRagType,BaseGraph>(ragModule, clsName.c_str());
        // export the rag shape
        clsT
            .def_property_readonly("shape",[](const GridRagType & self){return self.shape();})
        ;
        removeFunctions<GridRagType, BaseGraph>(clsT);

        // from labels
        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, DIM> labels,
               const int numberOfThreads
            ){
                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = numberOfThreads;
                ExplicitLabels<DIM, LABELS> explicitLabels(labels);
                auto ptr = new GridRagType(explicitLabels, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

        // from labels + serialization
        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, DIM>           labels,
               nifty::marray::PyView<uint64_t,   1, false>  serialization
            ){

                auto  startPtr = &serialization(0);
                auto  lastElement = &serialization(serialization.size()-1);
                auto d = lastElement - startPtr + 1;

                NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");


                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = -1;
                ExplicitLabels<DIM, LABELS> explicitLabels(labels);
                auto ptr = new GridRagType(explicitLabels,startPtr, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("serialization")
        );

    }

    #ifdef WITH_HDF5
    template<size_t DIM, class LABELS>
    void exportHdf5GridRagT(
        py::module & ragModule, 
        const std::string & clsName,
        const std::string & facName
    ){
        typedef UndirectedGraph<> BaseGraph;
        typedef Hdf5Labels<DIM, LABELS> LabelsProxyType;
        typedef GridRag<DIM, LabelsProxyType >  GridRagType;


        const auto labelsProxyClsName = clsName + std::string("LabelsProxy");
        const auto labelsProxyFacName = facName + std::string("LabelsProxy");
        py::class_<LabelsProxyType>(ragModule, labelsProxyClsName.c_str())
            .def("hdf5Array",&LabelsProxyType::hdf5Array,py::return_value_policy::reference)
        ;

        ragModule.def(labelsProxyFacName.c_str(),
            [](
               const hdf5::Hdf5Array<LABELS> & hdf5Array,
               const int64_t numberOfLabels
            ){
                auto ptr = new LabelsProxyType(hdf5Array, numberOfLabels);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("numberOfLabels")
        );



        auto clsT = py::class_<GridRagType, BaseGraph>(ragModule, clsName.c_str());
        clsT
            .def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
            .def_property_readonly("shape",[](const GridRagType & self){return self.shape();})
        ;

        removeFunctions<GridRagType, BaseGraph>(clsT);



        ragModule.def(facName.c_str(),
            [](
                const LabelsProxyType & labelsProxy,
                std::vector<int64_t>  blockShape,
                const int numberOfThreads
            ){
                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = numberOfThreads;

                if(blockShape.size() == DIM){
                    std::copy(blockShape.begin(), blockShape.end(), s.blockShape.begin());
                }
                else if(blockShape.size() == 1){
                    std::fill(s.blockShape.begin(), s.blockShape.end(), blockShape[0]);
                }
                else if(blockShape.size() != 0){
                    throw std::runtime_error("block shape has a non matching shape");
                }

                auto ptr = new GridRagType(labelsProxy, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labelsProxy"),
            py::arg_t< std::vector<int64_t>  >("blockShape", std::vector<int64_t>() ),
            py::arg_t< int >("numberOfThreads", -1 )
        );



        ragModule.def(facName.c_str(),
            [](
               const LabelsProxyType & labelsProxy,
               nifty::marray::PyView<uint64_t,   1, false>  serialization
            ){

                auto  startPtr = &serialization(0);
                auto  lastElement = &serialization(serialization.size()-1);
                auto d = lastElement - startPtr + 1;

                NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");


                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = -1;
    
                auto ptr = new GridRagType(labelsProxy, startPtr, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("serialization")
        );

    }
    #endif


    void exportGridRag(py::module & ragModule) {
        exportExpilictGridRagT<2, uint32_t>(ragModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint32_t>(ragModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");
        #ifdef WITH_HDF5
        exportHdf5GridRagT<2, uint32_t>(ragModule, "GridRagHdf5Labels2D", "gridRag2DHdf5");
        exportHdf5GridRagT<3, uint32_t>(ragModule, "GridRagHdf5Labels3D", "gridRag3DHdf5");
        #endif
    }
        

} // end namespace graph
} // end namespace nifty
