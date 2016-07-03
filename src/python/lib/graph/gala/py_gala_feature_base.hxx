#pragma once
#ifndef NIFTY_PYTHON_GRAPH_GALA_FEATURE_BASE_HXX
#define NIFTY_PYTHON_GRAPH_GALA_FEATURE_BASE_HXX

#include <string>
#include <initializer_list>

#include "nifty/graph/gala/gala_feature_base.hxx"

namespace nifty {
namespace graph {




template<class GRAPH, class T>
class PyGalaFeatureBase : public GalaFeatureBase<GRAPH, T> {
public:
    /* Inherit the constructors */
    // using MulticutFactory<Objective>::MulticutFactory;
    typedef GalaFeatureBase<GRAPH, T> BaseType;



    virtual uint64_t numberOfFeatures() const {
        PYBIND11_OVERLOAD_PURE(
            uint64_t,                
            BaseType,               
            numberOfFeatures                      
        );  
    }
    virtual void mergeEdges(const uint64_t alive, const uint64_t dead) {
        PYBIND11_OVERLOAD_PURE(
            void,                
            BaseType,               
            mergeEdges,
            alive, dead                     
        );  
    }
    virtual void mergeNodes(const uint64_t alive, const uint64_t dead) {
        PYBIND11_OVERLOAD_PURE(
            void,                
            BaseType,               
            mergeNodes,
            alive, dead                     
        ); 
    }
    virtual void getFeatures(const uint64_t edge, T * featuresOut){
        PYBIND11_OVERLOAD_PURE(
            void,                
            BaseType,               
            getFeatures,
            edge, featuresOut                     
        ); 
    }
    virtual void reset(){
        PYBIND11_OVERLOAD_PURE(
            void,                
            BaseType,               
            reset                    
        ); 
    }
};


} // namespace graph
} // namespace nifty

#endif /* NIFTY_PYTHON_GRAPH_GALA_FEATURE_BASE_HXX */
