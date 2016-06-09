#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "../../converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{



    void exportMulticutObjective(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> McObjective;
        const auto clsName = std::string("MulticutObjectiveUndirectedGraph");
        auto multicutObjectiveCls = py::class_<McObjective>(multicutModule, clsName.c_str());


        multicutModule.def("multicutObjective",
            [](const Graph & graph,  py::array_t<double> pyArray){
                NumpyArray<double> array(pyArray);
                NIFTY_CHECK_OP(array.dimension(),==,1,"wrong dimensions");
                NIFTY_CHECK_OP(array.shape(0),==,graph.numberOfEdges(),"wrong shape");
                
                auto obj = new McObjective(graph);
                auto & weights = obj->weights();
                graph.forEachEdge([&](int64_t edge){
                    weights[edge]=array(edge);
                });
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("graph"),py::arg("weights")  
        );

        //multicutObjectiveCls
        //    .def(py::init<const uint64_t,const uint64_t>()
        //      ,
        //       //py::arg("numberOfNodes"),
        //       py::arg("numberOfNodes"),
        //       py::arg_t<uint64_t>("reserveEdges",0)
        //    )
        //    //.def("insertEdges",
        //    //    [](Graph & g, py::array_t<uint64_t> pyArray) {
        //    //        NumpyArray<uint64_t> array(pyArray);
        //    //        NIFTY_CHECK_OP(array.dimension(),==,2,"wrong dimensions");
        //    //        NIFTY_CHECK_OP(array.shape(1),==,2,"wrong shape");
        //    //        for(size_t i=0; i<array.shape(0); ++i){
        //    //            g.insertEdge(array(i,0),array(i,1));
        //    //        }
        //    //    }
        //    //)
        //;


    }

}
}
