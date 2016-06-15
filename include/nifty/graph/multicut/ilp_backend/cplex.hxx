#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_ILP_BACKEND_CPLEX_HXX
#define NIFTY_GRAPH_MULTICUT_ILP_BACKEND_CPLEX_HXX

#include <limits>
#define IL_STD 1
#include <ilcplex/ilocplex.h>

#include "nifty/graph/multicut/ilp_backend/ilp_backend.hxx"

namespace nifty {
namespace graph {
namespace ilp_backend{
    
class Cplex {
public:
    typedef IlpBackendSettings Settings;

    Cplex(const Settings & settings = Settings());
    //~Cplex();

    void initModel(const size_t, const double*);

    template<class Iterator>
    void setStart(Iterator);

    template<class VariableIndexIterator, class CoefficientIterator>
    void addConstraint(VariableIndexIterator, VariableIndexIterator,CoefficientIterator, const double, const double);
    void optimize();
    double label(const size_t) const;


private:
   Settings         settings_;
   uint64_t         nVariables_;
   IloEnv           env_;
   IloModel         model_;
   IloNumVarArray   x_;
   IloRangeArray    c_;
   IloObjective     obj_;
   IloNumArray      sol_;
   IloCplex         cplex_;
};

inline Cplex::Cplex(const Settings & settings) 
:   settings_(settings),
    nVariables_(0),
    env_(),
    model_(),
    x_(),
    c_(),
    obj_(),
    sol_(),
    cplex_()
{
    
}




inline void
Cplex::initModel(
    const size_t numberOfVariables,
    const double* coefficients
) {

    try{


        nVariables_ = numberOfVariables;
        IloInt N = numberOfVariables;
        model_ = IloModel(env_);
        x_     = IloNumVarArray(env_);
        c_     = IloRangeArray(env_);
        obj_   = IloMinimize(env_);
        sol_   = IloNumArray(env_,N);
        // set variables and objective
        x_.add(IloNumVarArray(env_, N, 0, 1, ILOBOOL));

        IloNumArray    obj(env_,N);

        for(size_t v=0; v<numberOfVariables; ++v){
            obj[v] = coefficients[v];
        }
        obj_.setLinearCoefs(x_,obj);
        model_.add(obj_); 
        cplex_ = IloCplex(model_);


        // set the parameter
        if(settings_.absoluteGap >= 0.0)
            cplex_.setParam(IloCplex::EpAGap,settings_.absoluteGap);
        if(settings_.relativeGap >= 0.0)
            cplex_.setParam(IloCplex::EpGap,settings_.relativeGap);

        // nThreads
        cplex_.setParam(IloCplex::Threads, settings_.numberOfThreads);

        // verbosity
        cplex_.setParam(IloCplex::MIPDisplay, int(settings_.verbosity));
        cplex_.setParam(IloCplex::BarDisplay, int(settings_.verbosity));
        cplex_.setParam(IloCplex::SimDisplay, int(settings_.verbosity));
        cplex_.setParam(IloCplex::NetDisplay, int(settings_.verbosity));
        cplex_.setParam(IloCplex::SiftDisplay,int(settings_.verbosity));

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

    try{
        cplex_.getValues(sol_, x_);
    }
    catch (IloException& e) {
        std::cout<<"nVar == "<<nVariables_<<"\n";
        std::cout<<" error in get values "<<e<<" "<<e.getMessage()<<"\n";
        e.end();
    }
    catch (const std::runtime_error & e) {
        std::cout<<" error "<<e.what()<<"\n";
    }
    catch (const std::exception & e) {
        std::cout<<" error "<<e.what()<<"\n";
    }


    
}

inline double
Cplex::label(
    const size_t variableIndex
) const {
    return sol_[variableIndex];
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
    //std::cout<<"set start for n var "<<nVariables_<<"\n";
    //IloNumArray startVal(env_, nVariables_);
    for (auto i = 0; i < nVariables_; ++i) {
        //sol_[i] = *valueIterator;
        //std::cout<<i<<" x_ "<<x_[i]<<"\n";
        ++valueIterator;
    }
    
    //try{
    //cplex_.addMIPStart(x_, sol_);
    //}
    //catch (IloException& e) {
    //    std::cout<<" error "<<e.getMessage()<<"\n";
    //    e.end();
    //}
    //catch (const std::runtime_error & e) {
    //    std::cout<<" error "<<e.what()<<"\n";
    //}
    //catch (const std::exception & e) {
    //    std::cout<<" error "<<e.what()<<"\n";
    //}
    
}

} // namespace ilp_backend
} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_MULTICUT_ILP_BACKEND_CPLEX_HXX
