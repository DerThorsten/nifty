#pragma once

#include "nifty/python/graph/graph_name.hxx"
#include <boost/algorithm/string.hpp>
#include <sstream>

namespace nifty{
namespace graph{
namespace optimization{


    class SolverDocstringHelper{
    public:
        SolverDocstringHelper(){

        }

        template<class SOLVER>
        std::string solverDocstring()const{

            typedef typename SOLVER::GraphType GraphType;
            typedef typename SOLVER::ObjectiveType ObjectiveType;

            const auto graphClsName = GraphName<GraphType>::name();
            const auto graphModName = GraphName<GraphType>::moduleName();

            std::stringstream ss;
            ss<<upperFirst(objectiveName)<<" solver `"<<upperFirst(name)<<".`\n\n";

            // the main text from the user
            if(!mainText.empty()){
                ss<<mainText<<"\n\n";
            }

            // setup the related classes
            ss<<"See Also:\n\n";
            ss<<"    Objective Class: :class:`"<<objectiveClsName<<"`\n\n";
            ss<<"    Graph Class: :class:`"<<graphModName<<"."<<graphClsName<<"`\n\n";
            ss<<"    Solver Base Class: :class:`"<<solverBaseClsName<<"`\n\n";
            ss<<"    Solver Factory Class: :class:`"<<factoryClsName<<"`\n\n";
            //if(!seeAlso.empty()){
            //    for(const auto & sa: seeAlso){
            //        ss<<sa<<",";
            //    }
            //    ss<<"\n";
            //}
            // citations
            if(!cites.empty()){
                ss<<"**Cite:** ";
                for(const auto & cite: cites){
                    ss<<":cite:`"<<cite<<"` ";
                }
                ss<<"\n\n";
            }
            // notes
            if(!note.empty()){
                ss<<"Note:\n"<<insertTabs(note)<<"\n\n";
            }
            // warnings
            if(!warning.empty()){
                ss<<"Warning:\n\n"<<insertTabs(warning)<<"\n\n";
            }
            // example(s)
            if(!examples.empty()){
                if(examples.size()>=2)
                    ss<<"Examples:\n"<<"    ";
                else
                    ss<<"Example:\n"<<"    ";

                for(const auto & example: examples){
                    ss<<insertTabs(example)<<"\n\n";
                }
                ss<<"\n\n";
            }

            // see also
            // TODO, this should
            // never be empty
            // it should / can point to base classes
            if(!seeAlso.empty()){
                ss<<"See Also:\n"<<"    ";
                for(const auto & sa: seeAlso){
                    ss<<sa<<",";
                }
                ss<<"\n";
            }


            return ss.str();
        }

        template<class FACTORY>
        std::string factoryDocstring()const{

            
            typedef typename FACTORY::ObjectiveType     ObjectiveType;
            typedef typename ObjectiveType::GraphType   GraphType;

            const auto graphClsName = GraphName<GraphType>::name();
            const auto graphModName = GraphName<GraphType>::moduleName();

            std::stringstream ss;
            ss<<"Factory for "<<objectiveName<<" solver `"<<upperFirst(name)<<".`\n\n";

            
            ss<<"Factory class to create instances of :class:`"<<solverClsName<<"`\n\n";
            // setup the related classes
            ss<<"See Also:\n\n";
            ss<<"    Objective Class: :class:`"<<objectiveClsName<<"`\n\n";
            ss<<"    Graph Class: :class:`"<<graphModName<<"."<<graphClsName<<"`\n\n";
            ss<<"    Solver Class: :class:`"<<solverClsName<<"`\n\n";
            ss<<"    Factory Base Class: :class:`"<<factoryBaseClsName<<"`\n\n";


            return ss.str();
        }

        std::string mainText;
        std::string objectiveName;
        std::string objectiveClsName;
        std::string name;
        std::string clsName;
        std::vector<std::string> seeAlso;
        std::string warning;
        std::string note;
        std::vector<std::string> examples;

        std::vector<std::string> cites;
        std::vector<std::string> requirements;


        // autofilled
        std::string factoryBaseClsName;
        std::string solverBaseClsName;
        std::string solverClsName;
        std::string factoryClsName;
        std::string factoryFactoryName;
    private:

        std::string upperFirst(const std::string & name)const{
            auto ret = name;
            ret[0] = std::toupper(ret[0]);
            return ret;
        }

        std::string insertTabs(const std::string & rawString)const{
            auto ret = rawString;
            boost::replace_all(ret, "\n", "\n    ");
            return std::string("    ")+ret;
        }
    };






}
}
}
