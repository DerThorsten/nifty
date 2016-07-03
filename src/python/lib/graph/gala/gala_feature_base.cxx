#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/gala/gala_feature_base.hxx"

#include "../../converter.hxx"
#include "py_gala_feature_base.hxx"




namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(__T, std::shared_ptr<__T>);

namespace nifty{
namespace graph{



    using namespace py;
    //PYBIND11_DECLARE_HOLDER_TYPE(BASE_CLASS_, std::shared_ptr<BASE_CLASS_>);

    void exportGalaFeatureBase(py::module & galaModule) {

        typedef UndirectedGraph<> GraphType;
        typedef double FeatureValueType;
        typedef PyGalaFeatureBase<GraphType, FeatureValueType> PyGalaFeatureBaseType;
        typedef GalaFeatureBase<GraphType, FeatureValueType> GalaFeatureBaseType;

        //PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

        // base factory
        py::class_<
            GalaFeatureBaseType, 
            std::shared_ptr<GalaFeatureBaseType>, 
            PyGalaFeatureBaseType 
        > galaFeatureBase(galaModule, "GalaFeatureBaseTypeUndirectedGraph");
        
        galaFeatureBase
        ;

        
        // concrete visitors
        typedef DummyFeature<GraphType, FeatureValueType> GalaDummyFeature; 
        
        py::class_<GalaDummyFeature, std::shared_ptr<GalaDummyFeature> >(galaModule, "GalaDummyFeatureUndirectedGraph",  galaFeatureBase)
            .def(py::init<>())
        ;
        
    }

}
}
    
