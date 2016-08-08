#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#ifdef WITH_HDF52
#include "nifty/graph/rag/grid_rag_chunked.hxx"
#endif

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
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



        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), undirectedGraph);
        clsT
            .def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
        ;

        removeFunctions<GridRagType>(clsT);





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

    }


    template<class LABELS>
    void exportHdf5GridRagStacked2D(
        py::module & ragModule, 
        const std::string & clsName,
        const std::string & facName
    ){
        py::object baseGraphPyCls = ragModule.attr("GridRagHdf5Labels3D");
        
        typedef Hdf5Labels<3, LABELS> LabelsProxyType;
        typedef GridRagStacked2D<LabelsProxyType >  GridRagType;




        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), baseGraphPyCls);
        clsT
            .def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
            .def("minMaxLabelPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t, 2> out({size_t(shape[0]),size_t(2)});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    auto mima = self.minMaxNode(sliceIndex);
                    out(sliceIndex, 0) = mima.first;
                    out(sliceIndex, 1) = mima.second;
                }
                return out;
            })
            .def("numberOfNodesPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfNodes(sliceIndex);
                }
                return out;
            })
            .def("numberOfInSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("numberOfInBetweenSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInBetweenSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("inSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.inSliceEdgeOffset(sliceIndex);
                }
                return out;
            })
            .def("betweenSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.betweenSliceEdgeOffset(sliceIndex);
                }
                return out;
            })

        ;

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
            py::arg("labelsProxy"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

    }


    #endif


    void exportGridRag(py::module & ragModule, py::module & graphModule) {

        exportExpilictGridRagT<2, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");
        
        #ifdef WITH_HDF5
        exportHdf5GridRagT<2, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels2D", "gridRag2DHdf5");
        exportHdf5GridRagT<3, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels3D", "gridRag3DHdf5");
       
        exportHdf5GridRagStacked2D<uint32_t>(ragModule, "GridRagStacked2DHdf5", "gridRagStacked2DHdf5Impl");
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
