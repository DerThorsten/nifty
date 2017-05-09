#pragma once


#include "mincut_base.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MincutBase;


    template<class OBJECTIVE>
    class MincutFactoryBase{
    public:
        typedef OBJECTIVE Objective;
        typedef MincutBase<Objective> MincutBaseType;
        virtual ~MincutFactoryBase(){}
        virtual std::shared_ptr<MincutBaseType> createSharedPtr(const Objective & objective) = 0;
        virtual MincutBaseType * createRawPtr(const Objective & objective) = 0;
    };


    template<class SOLVER>
    class MincutFactory :
    public MincutFactoryBase<typename SOLVER::Objective>{
    public:
        typedef typename SOLVER::Objective Objective;
        typedef MincutBase<Objective> MincutBaseType;
        typedef SOLVER Solver;
        typedef typename Solver::Settings Settings;
        MincutFactory(const Settings & settings = Settings())
        :   MincutFactoryBase<Objective>(),
            options_(settings){
        }
        virtual std::shared_ptr<MincutBaseType> createSharedPtr(const Objective & objective){
            return std::make_shared<Solver>(objective, options_);
        }
        virtual MincutBaseType * createRawPtr(const Objective & objective){
            MincutBaseType *  p =  new Solver(objective, options_);
            return p;
        }
    private:
        Settings options_;
    };

} // namespace graph
} // namespace nifty

