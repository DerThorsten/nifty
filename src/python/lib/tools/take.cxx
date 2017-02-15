#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <typeinfo> // to debug atm

#include "nifty/python/converter.hxx"
#include "nifty/tools/make_dense.hxx"
#include "nifty/tools/timer.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T, bool AUTO_CAST>
    void exportTakeT(py::module & toolsModule) {


        toolsModule.def("_take",
        [](
           nifty::marray::PyView<T,1,AUTO_CAST> relabeling,
           nifty::marray::PyView<T,1,AUTO_CAST> toRelabel
        ){

            nifty::tools::VerboseTimer timer(true);
            timer.startAndPrint("out");
            nifty::marray::PyView<T> out(toRelabel.shapeBegin(), toRelabel.shapeEnd());
            timer.stopAndPrint().reset();


            {

                py::gil_scoped_release allowThreads;

                std::cout<<"simpel out        "<<out.isSimple()<<"\n";
                std::cout<<"simpel relabeling "<<relabeling.isSimple()<<"\n";
                std::cout<<"simpel toRelabel  "<<toRelabel.isSimple()<<"\n";


                timer.startAndPrint("work");
                for(size_t i=0; i<toRelabel.shape(0); ++i){
                    out(i) = relabeling(toRelabel(i));
                }




                //auto po = &out[0];
                //auto pr = &relabeling[0];
                //auto pt = & toRelabel[0];
                //for(size_t i=0; i<toRelabel.shape(0); ++i){
                //    po[i] = pr[pt[i]];
                //}

                timer.stopAndPrint().reset();
                
            }
            
            return out;
        });
    }


    void exportTake(py::module & toolsModule) {
        
        exportTakeT<uint32_t, false>(toolsModule);
        exportTakeT<uint64_t, false>(toolsModule);
        exportTakeT<int32_t, false>(toolsModule);

        //exportTakeT<float   , false>(toolsModule);

        exportTakeT<int64_t   , true>(toolsModule);

    }

}
}
