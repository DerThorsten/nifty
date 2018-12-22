#include <iostream>
#include <sstream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "xtensor-python/pyarray.hpp"

#include "nifty/ground_truth/overlap.hxx"
#include "nifty/ground_truth/partition_comparison.hxx"

namespace py = pybind11;


namespace nifty{
namespace ground_truth{




    void exportPartitionComparison(py::module & module){

        typedef VariationOfInformation<> ViType;
        py::class_<ViType>(module, "VariationOfInformation")
        .def(py::init([](xt::pyarray<uint32_t> labelsTruth,
                         xt::pyarray<uint32_t> labelsPrediction,
                         const bool ignoreDefaultLabel = false) {

                {
                    auto  startPtr = &labelsTruth(0);
                    auto  lastElement = &labelsTruth(labelsTruth.size()-1);
                    auto d = lastElement - startPtr + 1;
                    NIFTY_CHECK_OP(d,==,labelsTruth.size(),"labelsTruth must be contiguous")
                }
                {
                    auto  startPtr = &labelsPrediction(0);
                    auto  lastElement = &labelsPrediction(labelsPrediction.size()-1);
                    auto d = lastElement - startPtr + 1;
                    NIFTY_CHECK_OP(d,==,labelsPrediction.size(),"labelsPrediction must be contiguous")
                }

                return new ViType(&labelsTruth(0),
                                  &labelsTruth(0) + labelsTruth.size(),
                                  &labelsPrediction(0),
                                  ignoreDefaultLabel);
            }),
            py::arg("labelsTruth"),
            py::arg("labelsPrediction"),
            py::arg("ignoreDefaultLabel")=false
        )
        .def_property_readonly("value",&ViType::value)
        .def_property_readonly("valueFalseCut",&ViType::valueFalseCut)
        .def_property_readonly("valueFalseJoin",&ViType::valueFalseJoin)
        ;


        typedef RandError<> RandErrorType;
        py::class_<RandErrorType>(module, "RandError")
        .def(py::init([](xt::pyarray<uint32_t> labelsTruth,
                         xt::pyarray<uint32_t> labelsPrediction,
                         const bool ignoreDefaultLabel = false) {

                {
                    auto  startPtr = &labelsTruth(0);
                    auto  lastElement = &labelsTruth(labelsTruth.size()-1);
                    auto d = lastElement - startPtr + 1;
                    NIFTY_CHECK_OP(d,==,labelsTruth.size(),"labelsTruth must be contiguous")
                }
                {
                    auto  startPtr = &labelsPrediction(0);
                    auto  lastElement = &labelsPrediction(labelsPrediction.size()-1);
                    auto d = lastElement - startPtr + 1;
                    NIFTY_CHECK_OP(d,==,labelsPrediction.size(),"labelsPrediction must be contiguous")
                }

                return new RandErrorType(&labelsTruth(0),
                                         &labelsTruth(0) + labelsTruth.size(),
                                         &labelsPrediction(0),
                                         ignoreDefaultLabel);
            }),
            py::arg("labelsTruth"),
            py::arg("labelsPrediction"),
            py::arg("ignoreDefaultLabel")=false
        )
        .def_property_readonly("trueJoins",&RandErrorType::trueJoins)
        .def_property_readonly("trueCuts",&RandErrorType::trueCuts)
        .def_property_readonly("falseJoins",&RandErrorType::falseJoins)
        .def_property_readonly("falseCuts",&RandErrorType::falseCuts)
        .def_property_readonly("joinsInPrediction",&RandErrorType::joinsInPrediction)
        .def_property_readonly("cutsInPrediction",&RandErrorType::cutsInPrediction)
        .def_property_readonly("joinsInTruth",&RandErrorType::joinsInTruth)
        .def_property_readonly("cutsInTruth",&RandErrorType::cutsInTruth)

        .def_property_readonly("recallOfCuts",&RandErrorType::recallOfCuts)
        .def_property_readonly("precisionOfCuts",&RandErrorType::precisionOfCuts)
        .def_property_readonly("recallOfJoins",&RandErrorType::recallOfJoins)
        .def_property_readonly("precisionOfJoins",&RandErrorType::precisionOfJoins)
        .def_property_readonly("error",&RandErrorType::error)
        .def_property_readonly("index",&RandErrorType::index)

        ;
    }
}
}
