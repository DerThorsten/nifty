#pragma once
#ifndef NIFTY_ILP_GUROBI_HXX
#define NIFTY_ILP_GUROBI_HXX

#include <limits>
#include <string>

#include "gurobi_c++.h"

#include "nifty/graph/multicut/ilp_backend/ilp_backend.hxx"

namespace nifty {
namespace graph {
namespace ilp_backend{

class Gurobi {
public:

    typedef IlpBackendSettings Settings;

    Gurobi(const Settings & settings = Settings());
    ~Gurobi();

    void initModel(const size_t, const double*);

    template<class Iterator>
    void setStart(Iterator);

    template<class VariableIndexIterator, class CoefficientIterator>
        void addConstraint(VariableIndexIterator, VariableIndexIterator,
                           CoefficientIterator, const double, const double);
    void optimize();

    double label(const size_t) const;

    static std::string name(){
        return std::string("Gurobi");
    }

    template<class OBJECTIVE_ITERATOR>
    void changeObjective(
        OBJECTIVE_ITERATOR objectiveIter
    ){
        throw WeightsChangedNotSupported();

    }

private:
    Settings settings_;
    GRBEnv gurobiEnvironment_;
    GRBModel* gurobiModel_;
    GRBVar* gurobiVariables_;
    GRBLinExpr gurobiObjective_;
    size_t nVariables_;
};

inline Gurobi::Gurobi(const Settings & settings)
:   settings_(settings),
    gurobiEnvironment_(),
    gurobiModel_(NULL),
    gurobiVariables_(NULL),
    gurobiObjective_(),
    nVariables_(0)
{

}

inline
Gurobi::~Gurobi() {
    if (gurobiModel_ != NULL)
        delete gurobiModel_;

    if (gurobiVariables_ != NULL)
        delete[] gurobiVariables_;
}



inline void
Gurobi::initModel(
    const size_t numberOfVariables,
    const double* coefficients
) {
    nVariables_ = numberOfVariables;
    
    if (gurobiModel_ != NULL)
        delete gurobiModel_;

    if (gurobiVariables_ != NULL)
        delete[] gurobiVariables_;

    // verbosity
    gurobiEnvironment_.set(GRB_IntParam_OutputFlag, settings_.verbosity);

    // lp solver
    switch(settings_.lpSolver) {
        case Settings::LP_SOLVER_PRIMAL_SIMPLEX:
            gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 0);
            break;
        case Settings::LP_SOLVER_DUAL_SIMPLEX:
            gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 1);
            break;
        case Settings::LP_SOLVER_BARRIER:
            gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 2);
            break;
        case Settings::LP_SOLVER_SIFTING:
            gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 1); // dual simplex
            gurobiEnvironment_.set(GRB_IntParam_SiftMethod, 1); // moderate, 2 = aggressive
            break;
        default:
            break;
    }

    // number of threads
    gurobiEnvironment_.set(GRB_IntParam_Threads, settings_.numberOfThreads);

    
    // mip gaps
    if(settings_.absoluteGap >= 0.0)
        gurobiEnvironment_.set(GRB_DoubleParam_MIPGapAbs, settings_.absoluteGap);
    if(settings_.relativeGap >= 0.0)
        gurobiEnvironment_.set(GRB_DoubleParam_MIPGap, settings_.relativeGap);

    // presolver
    switch(settings_.preSolver) {
        case Settings::PRE_SOLVER_NONE:
            gurobiEnvironment_.set(GRB_IntParam_Presolve, 0);
            return;
        case Settings::PRE_SOLVER_AUTO:
            gurobiEnvironment_.set(GRB_IntParam_PreDual, -1);
            break;
        case Settings::PRE_SOLVER_PRIMAL:
            gurobiEnvironment_.set(GRB_IntParam_PreDual, 0);
            break;
        case Settings::PRE_SOLVER_DUAL:
            gurobiEnvironment_.set(GRB_IntParam_PreDual, 1);
            break;
        default:
            break;
    }
    if(settings_.prePasses > -1){
        gurobiEnvironment_.set(GRB_IntParam_PrePasses, settings_.prePasses);
    }
    

    // set the memory limit
    if(settings_.memLimit > 0.0)
        gurobiEnvironment_.set(GRB_DoubleParam_NodefileStart, settings_.memLimit);


    gurobiModel_ = new GRBModel(gurobiEnvironment_);
    gurobiVariables_ = gurobiModel_->addVars(numberOfVariables, GRB_BINARY);



    gurobiModel_->update();
    gurobiObjective_.addTerms(coefficients, gurobiVariables_, numberOfVariables);
    gurobiModel_->setObjective(gurobiObjective_);
}

inline void
Gurobi::optimize() {
    gurobiModel_->optimize();
}

inline double
Gurobi::label(
    const size_t variableIndex
) const {
    return gurobiVariables_[variableIndex].get(GRB_DoubleAttr_X);
}



template<class VariableIndexIterator, class CoefficientIterator>
inline void
Gurobi::addConstraint(
    VariableIndexIterator viBegin,
    VariableIndexIterator viEnd,
    CoefficientIterator coefficient,
    const double lowerBound,
    const double upperBound
) {
    GRBLinExpr expression;
    for(; viBegin != viEnd; ++viBegin, ++coefficient) {
        expression += (*coefficient) * gurobiVariables_[static_cast<size_t>(*viBegin)];
    }
    if(lowerBound == upperBound) {
        GRBLinExpr exact(lowerBound);
        gurobiModel_->addConstr(expression, GRB_EQUAL, exact);
    }
    else {
        if(lowerBound != -std::numeric_limits<double>::infinity()) {
            GRBLinExpr lower(lowerBound);
            gurobiModel_->addConstr(expression, GRB_GREATER_EQUAL, lower);
        }
        if(upperBound != std::numeric_limits<double>::infinity()) {
            GRBLinExpr upper(upperBound);
            gurobiModel_->addConstr(expression,GRB_LESS_EQUAL, upper);
        }
    }
}

template<class Iterator>
inline void
Gurobi::setStart(
    Iterator valueIterator
) {
    for(size_t j = 0; j < nVariables_; ++j, ++valueIterator) {
        gurobiVariables_[j].set(GRB_DoubleAttr_Start, static_cast<double>(*valueIterator));
    }
}

} // namespace ilp_backend
} // namespace graph
} // namespace nifty


#endif // #ifndef NIFTY_ILP_GUROBI_HXX
