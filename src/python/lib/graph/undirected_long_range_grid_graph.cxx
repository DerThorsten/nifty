
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <cstddef>
//#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_long_range_grid_graph.hxx"




namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;

    template<class CLS, class BASE>
    void removeFunctions(py::class_<CLS, BASE > & clsT){
        clsT
            .def("insertEdge", [](CLS * self,const uint64_t u,const uint64_t ){
                throw std::runtime_error("cannot insert edges into 'UndirectedLongRangeGridGraph'");
            })
            .def("insertEdges",[](CLS * self, py::array_t<uint64_t> pyArray) {
                throw std::runtime_error("cannot insert edges into 'UndirectedLongRangeGridGraph'");
            })
        ;
    }



    template<std::size_t DIM, class LABELS>
    void exportUndirectedLongRangeGridGraphT(
        py::module & ragModule
    ){
        typedef UndirectedGraph<> BaseGraph;
        typedef UndirectedLongRangeGridGraph<DIM> GraphType;

        const auto clsName = GraphName<GraphType>::name();
        auto clsT = py::class_<GraphType,BaseGraph>(ragModule, clsName.c_str());

        clsT.def(py::init([](
            std::array<int, DIM>    shape,
            xt::pytensor<int64_t, 2> offsets   
        ){
            typedef typename GraphType::OffsetVector OffsetVector;
            typedef typename GraphType::ShapeType ShapeType;

            ShapeType s;
            std::copy(shape.begin(), shape.end(), s.begin());
            NIFTY_CHECK_OP(offsets.shape()[1], == , DIM, "offsets has wrong shape");
            OffsetVector offsetVector(offsets.shape()[0]);
            for(auto i=0; i<offsetVector.size(); ++i){
                for(auto d=0; d<DIM; ++d){
                    offsetVector[i][d] = offsets(i, d);
                }
            }
            return new GraphType(s, offsetVector);
        }),
        py::arg("shape"),
        py::arg("offsets"))
        //
        .def("nodeFeatureDiffereces", [](
            const GraphType & g,
            xt::pytensor<float, DIM+1> nodeFeatures
        ){
            return g.nodeFeatureDiffereces(nodeFeatures);
        })
        .def("edgeValues", [](
            const GraphType & g,
            xt::pytensor<float, DIM+1> nodeFeatures
        ){
            return g.edgeValues(nodeFeatures);
        })

        .def("edgeOffsetIndex", [](const GraphType & g){
            return g.edgeOffsetIndex();
        })
        ;

        removeFunctions<GraphType, BaseGraph>(clsT);


    }





    void exportUndirectedLongRangeGridGraph(py::module & ragModule) {

        exportUndirectedLongRangeGridGraphT<2, uint32_t>(ragModule);
        exportUndirectedLongRangeGridGraphT<3, uint32_t>(ragModule);

  
    }


} // end namespace graph
} // end namespace nifty
