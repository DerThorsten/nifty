#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_ILP_BACKEND_CPLEX_HXX
#define NIFTY_GRAPH_OPTIMIZATION_MULTICUT_ILP_BACKEND_CPLEX_HXX

#include <limits>
#include <string>
#include <sstream>

#define IL_STD 1
#include <ilcplex/ilocplex.h>

#include "nifty/ilp_backend/ilp_backend.hxx"

namespace nifty {
namespace ilp_backend{
    
class Cplex {
public:
    typedef IlpBackendSettings Settings;

    Cplex(const Settings & settings = Settings());
    ~Cplex(){
        cplex_.end();
        env_.end();
    }

    

    void initModel(const size_t, const double*);

    template<class Iterator>
    void setStart(Iterator);

    template<class VariableIndexIterator, class CoefficientIterator>
    void addConstraint(VariableIndexIterator, VariableIndexIterator,CoefficientIterator, const double, const double);
    void optimize();
    double label(const size_t) const;

    static std::string name(){
        return std::string("Cplex");
    }
    
    template<class OBJECTIVE_ITERATOR>
    void changeObjective(
        OBJECTIVE_ITERATOR objectiveIter
    ){
        IloNumArray    obj(env_,nVariables_);
        for(size_t v=0; v<nVariables_; ++v){
            const auto val = *objectiveIter;

            if(std::abs(val)<=0.00000001){
                if(val<0.0){
                    obj[v] = -0.00000001;
                }
                else{
                    obj[v] =  0.00000001;
                }
            }
            else{
                obj[v] = val;
            }
            ++objectiveIter;
        }
        obj_.setLinearCoefs(x_,obj);
    }

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
    nVariables_ = numberOfVariables;
    if(nVariables_>=1){
        try{

            //std::cout<<"init cplex with "<<numberOfVariables<<" numberOfVariables\n";

            
            IloInt N = numberOfVariables;
            model_ = IloModel(env_);
            x_     = IloNumVarArray(env_);
            c_     = IloRangeArray(env_);
            obj_   = IloMinimize(env_);
            sol_   = IloNumArray(env_,N);
            // set variables and objective
            //std::cout<<"add var\n";
            x_.add(IloNumVarArray(env_, N, 0, 1, ILOBOOL));

            //std::cout<<"create obj\n";
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


            if(settings_.memLimit > 0.0)
                cplex_.setParam(IloCplex::TreLim,settings_.memLimit*1024.0);


            cplex_.setParam(IloCplex::Threads, settings_.numberOfThreads);
            // cplex_.setParam(IloCplex::EpAGap,0);
            // cplex_.setParam(IloCplex::EpGap,0);


            //cplex_.setParam(IloCplex::Threads, 1);
            cplex_.setParam(IloCplex::CutUp,  1.0e+75);
            cplex_.setParam(IloCplex::MIPDisplay, int(settings_.verbosity));
            cplex_.setParam(IloCplex::BarDisplay, int(settings_.verbosity));
            cplex_.setParam(IloCplex::SimDisplay, int(settings_.verbosity));
            cplex_.setParam(IloCplex::NetDisplay, int(settings_.verbosity));
            cplex_.setParam(IloCplex::SiftDisplay,int(settings_.verbosity));

            cplex_.setParam(IloCplex::EpOpt,1e-9);
            cplex_.setParam(IloCplex::EpRHS,1e-8); //setting this to 1e-9 seemed to be to agressive!
            cplex_.setParam(IloCplex::EpInt,0);




            // nThreads
            //

            // verbosity


        }
        catch (IloException& e) {
            const std::string msg = e.getMessage();
            e.end();
            throw std::runtime_error(std::string("error in initModel ")+msg);
        }
        catch (const std::runtime_error & e) {
            throw std::runtime_error(std::string("error in initModel ")+e.what());
        }
        catch (const std::exception & e) {
            throw std::runtime_error(std::string("error in initModel ")+e.what());
        }

    }




}

inline void
Cplex::optimize() {
    if(nVariables_>=1){
        try{
            if(!cplex_.solve()) {
                std::cout << "failed to optimize. " <<cplex_.getStatus() << std::endl;
                std::stringstream ss;
                ss<<"cplex failed with status: "<<cplex_.getStatus();
                throw std::runtime_error(ss.str());
            }
        }
        catch (IloException& e) {
            const std::string msg = e.getMessage();
            e.end();
            throw std::runtime_error(std::string("error in optimize: ")+msg);
        }
        catch (const std::runtime_error & e) {
            throw std::runtime_error(std::string("error in optimize ")+e.what());
        }
        catch (const std::exception & e) {
            throw std::runtime_error(std::string("error in optimize ")+e.what());
        }

        try{
            cplex_.getValues(sol_, x_);
        }
        catch (IloException& e) {
            const std::string msg = e.getMessage();
            e.end();
            throw std::runtime_error(std::string("error in get values: ")+msg);
        }
        catch (const std::runtime_error & e) {
            throw std::runtime_error(std::string("error in optimize ")+e.what());
        }
        catch (const std::exception & e) {
            throw std::runtime_error(std::string("error in optimize ")+e.what());
        }
        // std::cout<<"run opt done\n";
    }
    
}

inline double
Cplex::label(
    const size_t variableIndex
) const {
    if(nVariables_>=1){
        return sol_[variableIndex];
    }
    else{
        return 0.0;
    }
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
    if(nVariables_>=1){
        IloRange constraint(env_, lowerBound, upperBound);
        while(viBegin!=viEnd){

            constraint.setLinearCoef(x_[*viBegin], *coefficient);
            ++viBegin;
            ++coefficient;
        }
        model_.add(constraint);
    }

}

template<class Iterator>
inline void
Cplex::setStart(
    Iterator valueIterator
) {
    if(nVariables_>=1){
        for (auto i = 0; i < nVariables_; ++i) {
            sol_[i] = *valueIterator;
            //std::cout<<i<<" x_ "<<x_[i]<<" sol "<<sol_[i]<<"\n";
            ++valueIterator;
        }
        try{
            cplex_.addMIPStart(x_, sol_);
        }
        catch (IloException& e) {
            const std::string msg = e.getMessage();
            e.end();
            throw std::runtime_error(std::string("error in setStart: ")+msg);
        }
        catch (const std::runtime_error & e) {
           throw std::runtime_error(std::string("error in setStart ")+e.what());
        }
        catch (const std::exception & e) {
           throw std::runtime_error(std::string("error in setStart ")+e.what());
        }
    }
    
}

} // namespace ilp_backend
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_ILP_BACKEND_CPLEX_HXX
