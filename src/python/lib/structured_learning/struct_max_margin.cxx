#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>


#include "nifty/python/converter.hxx"

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
                StructMaxMarginOracleBase * oracle,
                const double lambda = 1.0,
                const double minEps = 1e-5,
                const uint64_t steps = 0,
                const bool optOracle = false,
                const bool verbose = true,
                const bool nonNegativeWeights = false
            ){
                typedef typename StructMaxMargin::OptimizerType Optimizer;
                typename  StructMaxMargin::Parameter p;
                p.optimizerParameter_.lambda = lambda;
                p.optimizerParameter_.min_eps = minEps;
                p.optimizerParameter_.steps = steps;
                p.optimizerParameter_.epsStrategy = optOracle ?Optimizer::EpsFromGap : Optimizer::EpsFromChange;
                p.optimizerParameter_.verbose_ = verbose;
                p.optimizerParameter_.nonNegativeWeights = nonNegativeWeights;

                auto obj = new StructMaxMargin(oracle, p);
                return obj;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("oracle"),
            py::arg("lambda") = 1.0,
            py::arg("minEps") = 1e-5,
            py::arg_t<uint64_t>("steps", 0) ,
            py::arg_t<bool>("optOracle", false) ,
            py::arg_t<bool>("verbose", false),
            py::arg_t<bool>("nonNegativeWeights", false)

        );




        structMaxMargin
            .def("learn",&StructMaxMargin::learn)
            .def("getWeights",[](const StructMaxMargin & self){
                const auto & weights = self.getWeights();

                nifty::marray::PyView<float> npW({weights.size()});

                for(size_t i=0; i<weights.size(); ++i){
                    npW[i] = weights[i];
                }

                return npW;
            })

        ;



    }
}
}