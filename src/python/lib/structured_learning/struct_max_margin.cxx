#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/structured_learning/learners/struct_max_margin/struct_max_margin.hxx"

namespace py = pybind11;


//PYBIND11_DECLARE_HOLDER_TYPE(SSMOracleBase, std::shared_ptr<SSMOracleBase>);

namespace nifty{
namespace structured_learning{





    class PyStructMaxMarginOracleBase : public StructMaxMarginOracleBase {
    public:
        /* Inherit the constructors */
        // using MulticutFactory<Objective>::MulticutFactory;



        /* Trampoline (need one for each virtual function) */
        void getGradientAndValue(const std::vector<double> & weights, double & value, std::vector<double> & gradients){
            PYBIND11_OVERLOAD_PURE(
                void,                                       /* Return type */
                StructMaxMarginOracleBase,                  /* Parent class */
                getGradientAndValue,                        /* Name of function */
                weights,  value, gradients                  /* Argument(s) */
            );
        }

        size_t numberOfWeights()const{
            PYBIND11_OVERLOAD_PURE(
                size_t,                                       /* Return type */
                StructMaxMarginOracleBase,                  /* Parent class */
                numberOfWeights,                        /* Name of function */
            );
        }
    };



    void exportStructMaxMargin(py::module & structuredLearningModule){


        typedef PyStructMaxMarginOracleBase PyOracleBase;
        typedef StructMaxMarginOracleBase OracleBase;

        
        // StructMaxMarginOracleBase
        py::class_<
            OracleBase, 
            std::unique_ptr<OracleBase>, 
            PyOracleBase 
        > oracleBase(structuredLearningModule, "StructMaxMarginOracleBase");


        // struct max margin itself
        py::class_<
            StructMaxMargin
        > structMaxMargin(structuredLearningModule, "StructMaxMargin")

        ;


        structuredLearningModule.def("structMaxMargin",
            []
            (
                StructMaxMarginOracleBase * oracle
            ){
                auto obj = new StructMaxMargin(oracle);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("oracle")
        );




        structMaxMargin.def("learn",&StructMaxMargin::learn)
        ;



    }
}
}