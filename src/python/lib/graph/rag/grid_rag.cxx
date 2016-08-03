#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"

#ifdef WITH_HDF52
#include "nifty/graph/rag/grid_rag_chunked.hxx"
#endif

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;
   
    template<class CLS>
    void removeFunctions(py::class_<CLS> & clsT){
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
        py::module & graphModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object undirectedGraph = graphModule.attr("UndirectedGraph");
        typedef ExplicitLabelsGridRag<DIM, LABELS> GridRagType;

        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), undirectedGraph);
        removeFunctions<GridRagType>(clsT);

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
    }

    #ifdef WITH_HDF5

    template<size_t DIM, class LABELS>
    void exportHdf5GridRagT(
        py::module & ragModule, 
        py::module & graphModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object undirectedGraph = graphModule.attr("UndirectedGraph");
        
        typedef Hdf5Labels<DIM, LABELS> LabelsProxyType;
        typedef GridRag<DIM, LabelsProxyType >  GridRagType;

        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), undirectedGraph);
        removeFunctions<GridRagType>(clsT);

        ragModule.def(facName.c_str(),
            [](
               const LabelsProxyType & labelsProxy,
               const int numberOfThreads
            ){
                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = numberOfThreads;
                auto ptr = new GridRagType(labelsProxy, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

    }

    #endif


    void exportGridRag(py::module & ragModule, py::module & graphModule) {

        exportExpilictGridRagT<2, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");
        
        #ifdef WITH_HDF5
        exportHdf5GridRagT<3, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels2D", "gridRagHdf5Labels2D");
        exportHdf5GridRagT<2, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels3D", "gridRagHdf5Labels3D");
        #endif


        // export ChunkedLabelsGridRagSliced
        #ifdef WITH_HDF52
        {
            py::object undirectedGraph = graphModule.attr("UndirectedGraph");
            typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRagSliced;

            auto clsT = py::class_<ChunkedLabelsGridRagSliced>(ragModule, "ChunkedLabelsGridRagSliced", undirectedGraph);
            removeFunctions<ExplicitLabelsGridRagType>(clsT);

            ragModule.def("chunkedLabelsGridRagSliced",
                [](const std::string & label_file,
                   const std::string & label_key,
                   const int numberOfThreads,
                   const bool lockFreeAlg 
                ){
                    auto s = typename  ChunkedLabelsGridRagSliced::Settings();
                    s.numberOfThreads = numberOfThreads;
                    s.lockFreeAlg = lockFreeAlg;

                    ChunkedLabels<3,uint32_t> chunkedLabels(label_file, label_key);
                    auto ptr = new ChunkedLabelsGridRagSliced(chunkedLabels, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0, 1>(),
                py::arg("label_file"),
                py::arg("label_key"),
                py::arg_t< int >("numberOfThreads", 1 ),
                py::arg_t< bool >("lockFreeAlg", false )
            );
        }
        #endif
    }
        

} // end namespace graph
} // end namespace nifty
