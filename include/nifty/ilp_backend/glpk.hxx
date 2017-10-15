#pragma once

#include <limits>
#include <string>

#include <glpk.h>   

#include "nifty/exceptions/exceptions.hxx"
#include "nifty/ilp_backend/ilp_backend.hxx"

namespace nifty {
namespace ilp_backend{

class Glpk {
public:

    typedef IlpBackendSettings SettingsType;

    Glpk(const SettingsType & settings = SettingsType());
    ~Glpk();

    void initModel(const size_t, const double*);

    template<class Iterator>
    void setStart(Iterator);

    template<class VariableIndexIterator, class CoefficientIterator>
        void addConstraint(VariableIndexIterator, VariableIndexIterator,
                           CoefficientIterator, const double, const double);
    void optimize();

    double label(const size_t) const;

    static std::string name(){
        return std::string("Glpk");
    }
    template<class OBJECTIVE_ITERATOR>
    void changeObjective(
        OBJECTIVE_ITERATOR objectiveIter
    ){
        throw exceptions::WeightsChangedNotSupported();
    }

private:
    SettingsType settings_;
    size_t nVariables_;

    glp_prob * lp;

    int addedConstraints_;

};

inline Glpk::Glpk(const SettingsType & settings)
:   settings_(settings),
    nVariables_(0),
    lp(nullptr),
    addedConstraints_(0)
{
    //std::cout<<"constructor \n";
}

inline
Glpk::~Glpk() {
    if(lp != nullptr){
        glp_delete_prob(lp);
        glp_free_env();
    }
}



inline void
Glpk::initModel(
    const size_t numberOfVariables,
    const double* coefficients
) {

    //std::cout<<"init model\n";
    nVariables_ = numberOfVariables;
    if(lp != nullptr){
        glp_delete_prob(lp);
        glp_free_env();
    }

    lp = glp_create_prob();
    //std::cout<<"add cols\n";
    glp_add_cols(lp, nVariables_);

    //std::cout<<"set coeffs\n";
    for(size_t i=0; i<nVariables_; ++i){
        glp_set_obj_coef(lp, i+1, coefficients[i]);

        // set bounds
        glp_set_col_bnds(lp, i+1, GLP_DB, 0, 1);
        glp_set_col_kind(lp, i+1, GLP_IV);
    }

    // settings
    glp_term_out(settings_.verbosity);
}

inline void
Glpk::optimize() {
    //glp_simplex(lp, NULL);
    glp_iocp parm;
    glp_init_iocp(&parm);


    if(settings_.relativeGap >= 0.0){
        parm.mip_gap = settings_.absoluteGap;
    }
    if(settings_.relativeGap >= 0.0){
        parm.mip_gap = settings_.absoluteGap;
    }

    if(settings_.preSolver != IlpBackendSettings::PRE_SOLVER_NONE){
        parm.presolve = GLP_ON;
        parm.binarize = GLP_ON;
    }
    int err = glp_intopt(lp, &parm);
}

inline double
Glpk::label(
    const size_t variableIndex
) const {
    //std::cout<<"get label\n";
    auto val = glp_mip_col_val  (lp, variableIndex+1);
    return val;
}



template<class VariableIndexIterator, class CoefficientIterator>
inline void
Glpk::addConstraint(
    VariableIndexIterator viBegin,
    VariableIndexIterator viEnd,
    CoefficientIterator coefficient,
    const double lowerBound,
    const double upperBound
) {
    //std::cout<<"add constraint "<< addedConstraints_<<" \n";
    glp_add_rows(lp ,1);
    glp_set_row_bnds(lp, addedConstraints_+1, GLP_DB, lowerBound, upperBound);

    //std::cout<<"nvar total "<<nVariables_<<"\n";


    const auto nVar = std::distance(viBegin, viEnd);
    //std::cout<<"var in const "<<nVar<<"\n";

    std::vector<int> indices(nVar+1);
    std::vector<double> coeffs(nVar+1);

    //indices.assign(viBegin, viEnd);
    for(size_t i=0; i<nVar; ++i){
        indices[i+1] = viBegin[i]+1;
        coeffs[i+1] = coefficient[i];
        //std::cout<<" "<<indices[i+1]<<"\n";
    }
    //std::vector<double> coeffs(coefficient,coefficient+indices.size());

    glp_set_mat_row(lp, addedConstraints_+1,nVar, indices.data(), coeffs.data());

    ++addedConstraints_;
}

template<class Iterator>
inline void
Glpk::setStart(
    Iterator valueIterator
) {

}

} // namespace ilp_backend
} // namespace nifty


