#pragma once
#ifndef NIFTY_ILP_GUROBI_HXX
#define NIFTY_ILP_GUROBI_HXX

#include <limits>

#include <ilcplex/ilocplex.h>

namespace nifty {
namespace ilp {

class Cplex {
public:
    enum PreSolver {PRE_SOLVER_AUTO, PRE_SOLVER_PRIMAL, PRE_SOLVER_DUAL, PRE_SOLVER_NONE};
    enum LPSolver {LP_SOLVER_PRIMAL_SIMPLEX, LP_SOLVER_DUAL_SIMPLEX, LP_SOLVER_BARRIER, LP_SOLVER_SIFTING};
    
    Cplex();
    ~Cplex();
    void setNumberOfThreads(const size_t);
    void setAbsoluteGap(const double);
    void setRelativeGap(const double);
    void setVerbosity(const bool);
    void setLPSolver(const LPSolver);
    void setPreSolver(const PreSolver, const int = -1);
    void initModel(const size_t, const double*);
    template<class Iterator>
        void setStart(Iterator);
    template<class VariableIndexIterator, class CoefficientIterator>
        void addConstraint(VariableIndexIterator, VariableIndexIterator,
                           CoefficientIterator, const double, const double);
    void optimize();

    double label(const size_t) const;
    size_t numberOfThreads() const;
    double absoluteGap() const;
    double relativeGap() const;

private:
   uint64_t       nVariables_;
   IloEnv         env_;
   IloModel       model_;
   IloNumVarArray x_;
   IloRangeArray  c_;
   IloObjective   obj_;
   IloNumArray    sol_;
   IloCplex       cplex_;
};

inline
Cplex::Cplex() 
{
    
}

inline
Cplex::~Cplex() {

}

inline void
Cplex::setNumberOfThreads(
    const size_t numberOfThreads
) {
    cplex_.setParam(IloCplex::Threads, numberOfThreads);
}

inline void
Cplex::setAbsoluteGap(
    const double gap
) {
    
}

inline void
Cplex::setRelativeGap(
    const double gap
) {
    
}

inline void
Cplex::setVerbosity(
    const bool verbosity
) {

   cplex_.setParam(IloCplex::MIPDisplay, int(verbosity));
   cplex_.setParam(IloCplex::BarDisplay, int(verbosity));
   cplex_.setParam(IloCplex::SimDisplay, int(verbosity));
   cplex_.setParam(IloCplex::NetDisplay, int(verbosity));
   cplex_.setParam(IloCplex::SiftDisplay,int(verbosity));
  
}

inline void
Cplex::setPreSolver(
    const PreSolver preSolver,
    const int passes
) {
   
}

inline void
Cplex::setLPSolver(
    const LPSolver lpSolver
) {

}

inline void
Cplex::initModel(
    const size_t numberOfVariables,
    const double* coefficients
) {
    IloInt N = numberOfVariables;
    model_ = IloModel(env_);
    x_     = IloNumVarArray(env_);
    c_     = IloRangeArray(env_);
    obj_   = IloMinimize(env_);
    sol_   = IloNumArray(env_,N);
    // set variables and objective
    x_.add(IloNumVarArray(env_, N, 0, 1, ILOFLOAT));

    IloNumArray    obj(env_,N);

    for(size_t v=0; v<numberOfVariables; ++v){
        obj[v] = coefficients[v];
    }
    obj_.setLinearCoefs(x_,obj);
    model_.add(obj_); 
    cplex_ = IloCplex(model_);
}

inline void
Cplex::optimize() {
    try{
        if(!cplex_.solve()) {
            std::cout << "failed to optimize. " <<cplex_.getStatus() << std::endl;
        }
    }
    catch (IloException& e) {
        std::cout<<" error "<<e.getMessage()<<"\n";
        e.end();
    }
    catch (const std::runtime_error & e) {
        std::cout<<" error "<<e.what()<<"\n";
    }
    catch (const std::exception & e) {
        std::cout<<" error "<<e.what()<<"\n";
    }
    cplex_.getValues(sol_, x_);
}

inline double
Cplex::label(
    const size_t variableIndex
) const {
    return sol_[variableIndex];
}

inline size_t
Cplex::numberOfThreads() const {
    return cplex_.getParam(IloCplex::Threads);
}

inline double
Cplex::absoluteGap() const {
    return 1.0;
}

inline double
Cplex::relativeGap() const {
    return 1.0;
}

template<class VariableIndexIterator, class CoefficientIterator>
inline void
Cplex::addConstraint(
    VariableIndexIterator viBegin,
    VariableIndexIterator viEnd,
    CoefficientIterator coefficient,
    const double lowerBound,
    const double upperBound
) {

    IloRange constraint(env_, lowerBound, upperBound);
    while(viBegin!=viEnd){
        constraint.setLinearCoef(x_[*viBegin], *coefficient);
        ++viBegin;
        ++coefficient;
    }
    model_.add(constraint);

}

template<class Iterator>
inline void
Cplex::setStart(
    Iterator valueIterator
) {
    
}

} // namespace ilp
} // namespace andres

#endif // #ifndef NIFTY_ILP_GUROBI_HXX
