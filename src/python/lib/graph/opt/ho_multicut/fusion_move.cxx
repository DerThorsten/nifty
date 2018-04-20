#include <pybind11/pybind11.h>




#include "nifty/graph/opt/ho_multicut/fusion_move.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/opt/ho_multicut/export_ho_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{

    template<class OBJECTIVE>
    void exportFusionMoveT(py::module & hoMulticutModule) {



        typedef OBJECTIVE ObjectiveType;
        const auto objName = HoMulticutObjectiveName<ObjectiveType>::name();
        const std::string factoryBaseName = std::string("hoMulticutFactoryBase")+objName;
        const std::string solverBaseName = std::string("hoMulticutBase") + objName;
        


        // the fusion mover parameter itself
        {
            typedef FusionMove<ObjectiveType> FusionMoveType;
            typedef typename FusionMoveType::SettingsType FusionMoveSettings;
            const auto fmSettingsName = std::string("__FusionMoveSettingsType") + objName;
            py::class_<FusionMoveSettings>(hoMulticutModule, fmSettingsName.c_str())
                .def(py::init<>())
                .def_readwrite("hoMcFactory",&FusionMoveSettings::hoMcFactory)
            ;

        }

    }

    void exportFusionMove(py::module & hoMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef HoMulticutObjective<GraphType, double> ObjectiveType;
            exportFusionMoveT<ObjectiveType>(hoMulticutModule);
        }
    }

} // namespace nifty::graph::opt::ho_multicut
} // namespace nifty::graph::opt
}
}
