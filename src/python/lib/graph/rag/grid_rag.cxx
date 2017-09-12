#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
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



    template<std::size_t DIM, class LABELS>
    void exportExpilictGridRagT(
        py::module & ragModule,
        const std::string & clsName,
        const std::string & facName
    ){
        typedef UndirectedGraph<> BaseGraph;
        typedef ExplicitLabelsGridRag<DIM, LABELS> GridRagType;

        auto clsT = py::class_<GridRagType,BaseGraph>(ragModule, clsName.c_str());
        removeFunctions<GridRagType, BaseGraph>(clsT);

        // from labels
        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, DIM> labels,
               const int numberOfThreads
            ){
                auto s = typename  GridRagType::SettingsType();
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
        // from labels
        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, DIM>           labels,
               nifty::marray::PyView<uint64_t,   1, false>  serialization
            ){

                auto  startPtr = &serialization(0);
                auto  lastElement = &serialization(serialization.size()-1);
                auto d = lastElement - startPtr + 1;

                NIFTY_CHECK_OP(d,==,serialization.size(), "serialization must be contiguous");


                auto s = typename  GridRagType::SettingsType();
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

    template<std::size_t DIM, class LABELS>
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
        ;

        removeFunctions<GridRagType, BaseGraph>(clsT);



        ragModule.def(facName.c_str(),
            [](
                const LabelsProxyType & labelsProxy,
                std::vector<int64_t>  blockShape,
                const int numberOfThreads
            ){
                auto s = typename  GridRagType::SettingsType();
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


                auto s = typename  GridRagType::SettingsType();
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

    template<class LABELS>
    void exportHdf5GridRagStacked2D(
        py::module & ragModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object baseGraphPyCls = ragModule.attr("GridRagHdf5Labels3D");




        typedef Hdf5Labels<3, LABELS> LabelsProxyType;
        typedef GridRag<3, LabelsProxyType >  BaseGraph;
        typedef GridRagStacked2D<LabelsProxyType >  GridRagType;




        auto clsT = py::class_<GridRagType, BaseGraph>(ragModule, clsName.c_str());
        clsT
            .def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
            .def("minMaxLabelPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t, 2> out({std::size_t(shape[0]),std::size_t(2)});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    auto mima = self.minMaxNode(sliceIndex);
                    out(sliceIndex, 0) = mima.first;
                    out(sliceIndex, 1) = mima.second;
                }
                return out;
            })
            .def("numberOfNodesPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfNodes(sliceIndex);
                }
                return out;
            })
            .def("numberOfInSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("numberOfInBetweenSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInBetweenSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("inSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.inSliceEdgeOffset(sliceIndex);
                }
                return out;
            })
            .def("betweenSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.betweenSliceEdgeOffset(sliceIndex);
                }
                return out;
            })

        ;

        removeFunctions<GridRagType, BaseGraph>(clsT);

        ragModule.def(facName.c_str(),
            [](
                const LabelsProxyType & labelsProxy,
                const int numberOfThreads
            ){
                auto s = typename  GridRagType::SettingsType();
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


    template<class LABELS>
    void exportExplicitGridRagStacked2D(
        py::module & ragModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object baseGraphPyCls = ragModule.attr("ExplicitLabelsGridRag3D");

        typedef ExplicitLabels<3, LABELS> LabelsProxyType;
        typedef GridRag<3, LabelsProxyType >  BaseGraph;
        typedef GridRagStacked2D<LabelsProxyType >  GridRagType;



        auto clsT = py::class_<GridRagType, BaseGraph>(ragModule, clsName.c_str());
        clsT
            //.def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
            .def("minMaxLabelPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t, 2> out({std::size_t(shape[0]),std::size_t(2)});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    auto mima = self.minMaxNode(sliceIndex);
                    out(sliceIndex, 0) = mima.first;
                    out(sliceIndex, 1) = mima.second;
                }
                return out;
            })
            .def("numberOfNodesPerSlice",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfNodes(sliceIndex);
                }
                return out;
            })
            .def("numberOfInSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("numberOfInBetweenSliceEdges",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.numberOfInBetweenSliceEdges(sliceIndex);
                }
                return out;
            })
            .def("inSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.inSliceEdgeOffset(sliceIndex);
                }
                return out;
            })
            .def("betweenSliceEdgeOffset",[](const GridRagType & self){
                const auto & shape = self.shape();
                nifty::marray::PyView<uint64_t,  1> out({std::size_t(shape[0])});
                for(auto sliceIndex = 0; sliceIndex<shape[0]; ++sliceIndex){
                    out(sliceIndex) =  self.betweenSliceEdgeOffset(sliceIndex);
                }
                return out;
            })

        ;

        removeFunctions<GridRagType, BaseGraph>(clsT);



        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, 3> labels,
               const int numberOfThreads
            ){
                auto s = typename  GridRagType::SettingsType();
                s.numberOfThreads = numberOfThreads;
                ExplicitLabels<3, LABELS> explicitLabels(labels);
                auto ptr = new GridRagType(explicitLabels, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg_t< int >("numberOfThreads", -1 )
        );

    }




    void exportGridRag(py::module & ragModule) {

        exportExpilictGridRagT<2, uint8_t>(ragModule, "ExplicitLabelsGridRag2D_uint8", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint8_t>(ragModule, "ExplicitLabelsGridRag3D_uint8", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, uint16_t>(ragModule, "ExplicitLabelsGridRag2D_uint16", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint16_t>(ragModule, "ExplicitLabelsGridRag3D_uint16", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, uint32_t>(ragModule, "ExplicitLabelsGridRag2D_uint32", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint32_t>(ragModule, "ExplicitLabelsGridRag3D_uint32", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, uint64_t>(ragModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint64_t>(ragModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, int8_t>(ragModule, "ExplicitLabelsGridRag2D_int8", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, int8_t>(ragModule, "ExplicitLabelsGridRag3D_int8", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, int16_t>(ragModule, "ExplicitLabelsGridRag2D_int16", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, int16_t>(ragModule, "ExplicitLabelsGridRag3D_int16", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, int32_t>(ragModule, "ExplicitLabelsGridRag2D_int32", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, int32_t>(ragModule, "ExplicitLabelsGridRag3D_int32", "explicitLabelsGridRag3D");

        exportExpilictGridRagT<2, int64_t>(ragModule, "ExplicitLabelsGridRag2D_int64", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, int64_t>(ragModule, "ExplicitLabelsGridRag3D_int64", "explicitLabelsGridRag3D");

        exportExplicitGridRagStacked2D<uint32_t>(ragModule, "GridRagStacked2DExplicit", "gridRagStacked2DExplicitImpl");

        #ifdef WITH_HDF5
        exportHdf5GridRagT<2, uint32_t>(ragModule, "GridRagHdf5Labels2D", "gridRag2DHdf5");
        exportHdf5GridRagT<3, uint32_t>(ragModule, "GridRagHdf5Labels3D", "gridRag3DHdf5");

        exportHdf5GridRagStacked2D<uint32_t>(ragModule, "GridRagStacked2DHdf5", "gridRagStacked2DHdf5Impl");
        #endif
    }


} // end namespace graph
} // end namespace nifty
