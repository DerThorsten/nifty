
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include <iostream>

#include "nifty/pipelines/neuro_seg.hxx"
#include "nifty/tools/blocking.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace pipelines{
namespace neuro_seg{

    void exporBlockData( py::module & mod){

        py::class_<BlockData>(mod,"BlockData")

            .def("__init__",
                [](
                    BlockData & instance,
                    typename BlockData::BlockingType & blocking,
                    const size_t blockIndex,
                    const uint8_t numberOfChannels, 
                    const uint8_t numberOfBins

                ){
                    {
                        py::gil_scoped_release allowThreads;
                        new (&instance) BlockData(blocking, blockIndex, numberOfChannels, numberOfBins);
                    }
                },
                py::arg("blocking"),
                py::arg("blockIndex"),
                py::arg("numberOfChannels"),
                py::arg("numberOfBins")
            )
            .def("accumulate",[]
                (
                    BlockData & instance,
                    marray::PyView<uint32_t, 3> labels,
                    marray::PyView<float,    4> data
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        instance.accumulate(labels, data);
                    }
                }
            )
            .def("merge",[]
                (
                    BlockData & instance,
                    const BlockData & other
                ){
                    {
                        py::gil_scoped_release allowThreads;
                        instance.merge(other);
                    }
                }
            )
            .def("maxNodeId", &BlockData::maxNodeId)


            .def("edgeCounts",[](
                const BlockData & instance
            ){  
                marray::PyView<float> out({instance.numberOfEdges()});
                {
                    py::gil_scoped_release allowThreads;
                    instance.edgeCounts(out);
                }
                return out;
            })
            .def("edgeMeans",[](
                const BlockData & instance,
                const size_t channel
            ){  
                marray::PyView<float> out({instance.numberOfEdges()});
                {
                    py::gil_scoped_release allowThreads;
                    instance.edgeMeans(channel, out);
                }
                return out;
            })

            .def("nodeCounts",[](
                const BlockData & instance
            ){  
                marray::PyView<float> out({instance.maxNodeId() + 1});
                {
                    py::gil_scoped_release allowThreads;
                    instance.nodeCounts(out);
                }
                return out;
            })
            .def("nodeMeans",[](
                const BlockData & instance,
                const size_t channel
            ){  
                marray::PyView<float> out({instance.maxNodeId() + 1});
                {
                    py::gil_scoped_release allowThreads;
                    instance.nodeMeans(channel, out);
                }
                return out;
            })



            .def("uvIds",[](
                const BlockData & instance
            ){  
                marray::PyView<uint32_t> uv({instance.numberOfEdges(), size_t(2)});
                {
                    py::gil_scoped_release allowThreads;
                    instance.uvIds(uv);
                }

                return uv;
            })

             // features for single edge
            .def("extractFeatures",[](
                BlockData & instance,
                const uint32_t u,
                const uint32_t v
            ){  
                // make it just large enough atm //FIXME
                marray::PyView<float> f({instance.numberOfFeatures()});
                {
                    py::gil_scoped_release allowThreads;
                    instance.extractFeatures(Edge(u,v), f);
                }
                return f;

            })

             // features for some edges
            .def("extractFeatures",[](
                BlockData & instance,
                marray::PyView<uint32_t, 2> uvIds
            ){  
                NIFTY_CHECK_OP(uvIds.shape(1),==,2,"uvIds has wrong shape");
                const size_t n = uvIds.shape(0);


                marray::PyView<float> f({n, instance.numberOfFeatures()});
                {
                    py::gil_scoped_release allowThreads;

                    for(auto i=0; i<n; ++i){

                        const auto edge = Edge(uvIds(i,0), uvIds(i,1));
                        auto fSub  = f.boundView(0, i);
                        instance.extractFeatures(edge, fSub);
                    }

                    
                }
                return f;

            })

            // features for all edges
            .def("extractFeatures",[](
                BlockData & instance
            ){  

                marray::PyView<float> f({instance.numberOfEdges(), instance.numberOfFeatures()});
                {
                    py::gil_scoped_release allowThreads;

                    instance.extractFeatures(f);
                    
                }
                return f;

            })

            ;
        ;
    }  

}
}
}




PYBIND11_PLUGIN(_neuro_seg) {
    py::module mod("_neuro_seg", "neuro seg submodule of nifty");

    using namespace nifty::pipelines::neuro_seg;


    exporBlockData(mod);
        
    return mod.ptr();
}

