#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "nifty/python/converter.hxx"
#include "nifty/tools/changable_priority_queue.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{



    void exportChangeablePriorityQueue(py::module & toolsModule){

        typedef ChangeablePriorityQueue<double> QueueType;
        typedef typename QueueType::priority_type priority_type;
        typedef typename QueueType::ValueType ValueType;
        typedef typename QueueType::const_reference const_reference;



        const auto clsStr = std::string("ChangeablePriorityQueue");
        py::class_<QueueType>(toolsModule, clsStr.c_str())

            .def(py::init([](const std::size_t maxSize){
                    return new QueueType(maxSize);
                })
            )

            .def("__len__",&QueueType::size)
            .def("__contains__",&QueueType::contains)
            .def("__delitem__",&QueueType::deleteItem)

            .def("reset",&QueueType::reset)
            .def("empty",&QueueType::empty)
            .def("clear",&QueueType::clear)

            .def("push",&QueueType::push)
            .def("top",&QueueType::top)
            .def("pop",&QueueType::pop)
            .def("topPriority",&QueueType::topPriority)

            .def("deleteItem",&QueueType::deleteItem)
            .def("changePriority",&QueueType::changePriority)
        ;
    }


}
}
