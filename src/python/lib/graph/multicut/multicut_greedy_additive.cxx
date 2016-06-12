#include <pybind11/pybind11.h>

#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_greedy_additive.hxx"

#include "../../converter.hxx"
#include "py_multicut_factory.hxx"
#include "py_multicut_base.hxx"



namespace py = pybind11;

//PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{




    void exportMulticutGreedyAdditive(py::module & multicutModule) {


        py::object factoryBase = multicutModule.attr("MulticutFactoryBaseUndirectedGraph");
        py::object solverBase = multicutModule.attr("MulticutBaseUndirectedGraph");

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> Objective;
        typedef PyMulticutFactoryBase<Objective> PyMcFactoryBase;
        typedef MulticutFactoryBase<Objective> McFactoryBase;

        typedef PyMulticutBase<Objective> PyMcBase;
        typedef MulticutBase<Objective> McBase;




            
        typedef MulticutGreedyAdditive<Objective> Solver;
        typedef typename Solver::Settings Settings;
        typedef MulticutFactory<Solver> Factory;

        // settings
        py::class_< Settings >(multicutModule, "MulticutGreedyAdditiveSettingsUndirectedGraph")
            .def(py::init<>())
            .def_readwrite("nodeNumStopCond", &Settings::nodeNumStopCond)
            .def_readwrite("weightStopCond", &Settings::weightStopCond)
            .def_readwrite("verbose", &Settings::verbose)
        ;

        // solver
        py::class_<Solver,std::shared_ptr<McBase> >(multicutModule, "MulticutGreedyAdditiveUndirectedGraph",  solverBase)
            //.def(py::init<>())
        ;

        // factory
        py::class_<Factory>(multicutModule, "MulticutGreedyAdditiveFactoryUndirectedGraph",  factoryBase)
            .def(py::init<const Settings &>(),
                py::arg_t<Settings>("setttings",Settings())
            )
        ;


        



    }

}
}
