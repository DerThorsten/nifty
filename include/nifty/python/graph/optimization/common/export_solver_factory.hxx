#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"



#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/common/py_solver_factory_base.hxx"





namespace nifty{
namespace graph{
namespace optimization{
namespace common{

    template<
        class SOLVER_BASE
    >
    void exportSolverFactory(
        py::module & module,
        const std::string objectiveName
    ){

        typedef SOLVER_BASE SolverBaseType;
        typedef typename SolverBaseType::ObjectiveType  ObjectiveType;
        
        typedef PySolverFactoryBase<SolverBaseType> PySolverFactoryBaseType;
        typedef typename PySolverFactoryBaseType::BaseType SolverFactoryBaseType;



        const auto clsName = std::string("SolverFactoryBase") + objectiveName;

        // base factory
        pybind11::class_<
            SolverFactoryBaseType, 
            std::shared_ptr<SolverFactoryBaseType>, 
            PySolverFactoryBaseType 
        > solverFactoryBase(module, clsName.c_str());
        
        solverFactoryBase
            .def(pybind11::init<>())

            .def("create", 
                [](SolverFactoryBaseType * self, const ObjectiveType & obj){
                    return self->create(obj);
                },
                pybind11::return_value_policy::take_ownership,
                pybind11::keep_alive<0,2>()
                )
        ;

    }



} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namepsace nifty
