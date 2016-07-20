#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #include <hdf5_hl.h>    
// #include <hdf5.h>



#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

    void exportGridRag(py::module & ragModule, py::module & graphModule) {


        // export ExplicitLabelsGridRag2D
        {
            py::object undirectedGraph = graphModule.attr("UndirectedGraph");
            typedef ExplicitLabelsGridRag<2, uint32_t> ExplicitLabelsGridRag2D;

            py::class_<ExplicitLabelsGridRag2D>(ragModule, "ExplicitLabelsGridRag2D", undirectedGraph)
                // remove a few methods
                .def("insertEdge", [](ExplicitLabelsGridRag2D * self,const uint64_t u,const uint64_t ){
                    throw std::runtime_error("cannot insert edges into 'ExplicitLabelsGridRag'");
                })
                .def("insertEdges",[](ExplicitLabelsGridRag2D * self, py::array_t<uint64_t> pyArray) {
                    throw std::runtime_error("cannot insert edges into 'ExplicitLabelsGridRag'");
                })
            ;
            ragModule.def("explicitLabelsGridRag2D",
                [](
                   nifty::marray::PyView<uint32_t, 2> labels,
                   const int numberOfThreads,
                   const bool lockFreeAlg 
                ){
                    auto s = typename  ExplicitLabelsGridRag2D::Settings();
                    s.numberOfThreads = numberOfThreads;
                    s.lockFreeAlg = lockFreeAlg;
                    ExplicitLabels<2, uint32_t> explicitLabels(labels);
                    auto ptr = new ExplicitLabelsGridRag2D(explicitLabels, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0, 1>(),
                py::arg("labels"),
                py::arg_t< int >("numberOfThreads", -1 ),
                py::arg_t< bool >("lockFreeAlg", false )
            );
        }
        
        // export ExplicitLabelsGridRag3D
        {
            py::object undirectedGraph = graphModule.attr("UndirectedGraph");
            typedef ExplicitLabelsGridRag<3, uint32_t> ExplicitLabelsGridRag3D;

            py::class_<ExplicitLabelsGridRag3D>(ragModule, "ExplicitLabelsGridRag3D", undirectedGraph)
                // remove a few methods
                .def("insertEdge", [](ExplicitLabelsGridRag3D * self,const uint64_t u,const uint64_t ){
                    throw std::runtime_error("cannot insert edges into 'ExplicitLabelsGridRag'");
                })
                .def("insertEdges",[](ExplicitLabelsGridRag3D * self, py::array_t<uint64_t> pyArray) {
                    throw std::runtime_error("cannot insert edges into 'ExplicitLabelsGridRag'");
                })
            ;
            ragModule.def("explicitLabelsGridRag3D",
                [](nifty::marray::PyView<uint32_t, 3> labels,
                   const int numberOfThreads,
                   const bool lockFreeAlg 
                ){
                    auto s = typename  ExplicitLabelsGridRag3D::Settings();
                    s.numberOfThreads = numberOfThreads;
                    s.lockFreeAlg = lockFreeAlg;
                    ExplicitLabels<3 ,uint32_t> explicitLabels(labels);
                    auto ptr = new ExplicitLabelsGridRag3D(explicitLabels, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0, 1>(),
                py::arg("labels"),
                py::arg_t< int >("numberOfThreads", -1 ),
                py::arg_t< bool >("lockFreeAlg", false )
            );
        }
        
        // export ChunkedLabelsGridRagSliced
        {
            py::object undirectedGraph = graphModule.attr("UndirectedGraph");
            typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRagSliced;

            py::class_<ChunkedLabelsGridRagSliced>(ragModule, "ChunkedLabelsGridRagSliced", undirectedGraph)
                // remove a few methods
                .def("insertEdge", [](ChunkedLabelsGridRagSliced * self,const uint64_t u,const uint64_t ){
                    throw std::runtime_error("cannot insert edges into 'ChunkedLabelsGridRagSliced'");
                })
                .def("insertEdges",[](ChunkedLabelsGridRagSliced * self, py::array_t<uint64_t> pyArray) {
                    throw std::runtime_error("cannot insert edges into 'ChunkedLabelsGridRagSliced'");
                })
            ;
            //TODO give chunk shape as parameter ?
            ragModule.def("chunkedLabelsGridRagSliced",
                [](const std::string & label_file,
                   const std::string & label_key,
                   const int numberOfThreads,
                   const bool lockFreeAlg 
                ){
                    auto s = typename  ChunkedLabelsGridRagSliced::Settings();
                    s.numberOfThreads = numberOfThreads;
                    s.lockFreeAlg = lockFreeAlg;

                    // get the array shape
                    //vigra::HDF5File file(label_file, vigra::HDF5File::ReadOnly);
                    //vigra::ChunkedArrayHDF5<3,uint32_t> labels(file, label_key );//temp(file, label_key );
                    
                    //auto shape = temp.shape();
                    //// construct array witht the correct chunk shape
                    //vigra::Shape3 chunk_shape(1,512,512);
                    //vigra::ChunkedArrayHDF5<3,uint32_t> labels(file, label_key, vigra::HDF5File::ReadOnly, shape, chunk_shape );
                    
                    ChunkedLabels<3,uint32_t> chunkedLabels(label_file, label_key);
                    auto ptr = new ChunkedLabelsGridRagSliced(chunkedLabels, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                //py::keep_alive<0, 1>(),
                py::arg("label_file"),
                py::arg("label_key"),
                py::arg_t< int >("numberOfThreads", -1 ),
                py::arg_t< bool >("lockFreeAlg", false )
            );
        }
    }

} // end namespace graph
} // end namespace nifty
    
