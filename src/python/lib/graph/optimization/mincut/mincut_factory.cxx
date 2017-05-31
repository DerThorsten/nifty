#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"


#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/mincut/py_mincut_factory.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{



    template<class OBJECTIVE>
    void exportMincutFactoryT(py::module & mincutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef PyMincutFactoryBase<ObjectiveType> PyMcFactoryBase;
        typedef MincutFactoryBase<ObjectiveType> McFactoryBase;


        const auto objName = MincutObjectiveName<ObjectiveType>::name();
        const auto clsName = std::string("MincutFactoryBase") + objName;

        // base factory
        py::class_<
            McFactoryBase, 
            std::shared_ptr<McFactoryBase>, 
            PyMcFactoryBase 
        > mcFactoryBase(mincutModule, clsName.c_str());
        
        mcFactoryBase
            .def(py::init<>())

            .def("create", 
                //&McFactoryBase::create,
                [](McFactoryBase * self, const ObjectiveType & obj){
                    return self->createRawPtr(obj);
                },
                //,
                py::return_value_policy::take_ownership,
                py::keep_alive<0,2>()
                )
        ;

    }

    void exportMincutFactory(py::module & mincutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutFactoryT<ObjectiveType>(mincutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            exportMincutFactoryT<ObjectiveType>(mincutModule);
        }
    }

} // namespace nifty::graph::optimization::mincut    
} // namespace nifty::graph::optimization
}
}
