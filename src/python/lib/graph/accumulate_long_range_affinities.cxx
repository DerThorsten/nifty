#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "xtensor-python/pytensor.hpp"



#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;


    // -----------------------------------------------------------------------------------------------
    // Version with the additional features:
    //  - does not require a RAG but simply a graph and a label image (can include long-range edges)
    //  - can perform weighted average of affinities depending on given affinitiesWeights
    //  - ignore pixels with ignore label

    template<std::size_t DIM, class GRAPH, class LABELS_TYPE>
    void exportAccumulateAffinitiesMeanAndLength(
            py::module & module
    ) {
        module.def("accumulateAffinitiesMeanAndLength_impl_",
                      [](
                              const GRAPH &graph,
                              xt::pytensor<LABELS_TYPE, DIM> labels,
                              xt::pytensor<float, DIM + 1> affinities,
                              xt::pytensor<float, DIM + 1> affinitiesWeights,
                              xt::pytensor<int, 2> offsets,
                              const bool hasIgnoreLabel,
                              const uint64_t ignoreLabel,
                              const int numberOfThreads
                      ) {
                          array::StaticArray<uint64_t, DIM> shape;

                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape()[d];
                              NIFTY_CHECK_OP(shape[d],==,affinities.shape()[d], "affinities have wrong shape");
                              NIFTY_CHECK_OP(shape[d],==,affinitiesWeights.shape()[d], "affinities have wrong shape");
                          }
                          NIFTY_CHECK_OP(offsets.shape()[0],==,affinities.shape()[DIM], "Affinities and offsets do not match");
                          NIFTY_CHECK_OP(offsets.shape()[0],==,affinitiesWeights.shape()[DIM], "Affinities and offsets do not match");


                          // Create thread pool:
                          nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                          nifty::parallel::ThreadPool threadpool(pOpts);
                          const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                          // Create temporary arrays
                          const std::size_t nb_edges = uint64_t(graph.edgeIdUpperBound()+1);
                          xt::pytensor<float, 2> accAff = xt::zeros<float>({actualNumberOfThreads, nb_edges});
                          xt::pytensor<float, 2> maxAff = xt::zeros<float>({actualNumberOfThreads, nb_edges});
                          xt::pytensor<float, 2> counter = xt::zeros<float>({actualNumberOfThreads, nb_edges});
                          {
                              py::gil_scoped_release allowThreads;
                              nifty::tools::parallelForEachCoordinate(threadpool,
                                                                      shape,
                                                                      [&](const auto threadId, const auto & coordP){
                                                                          const auto u = labels[coordP];
                                                                          if ((!hasIgnoreLabel) || (hasIgnoreLabel && u!=ignoreLabel)) {
                                                                              for(auto i=0; i<offsets.shape()[0]; ++i){
                                                                                  auto coordQ = coordP;
                                                                                  for (auto d=0; d<DIM; ++d) {
                                                                                      coordQ[d] += offsets(i,d);
                                                                                  }
                                                                                  if(coordQ.allInsideShape(shape)){
                                                                                      const auto v = labels[coordQ];
                                                                                      // Check if there is ignore label:
                                                                                      if ((!hasIgnoreLabel) || (hasIgnoreLabel && v!=ignoreLabel)) {
                                                                                          // Only consider if pixels are in different clusters:
                                                                                          if (u != v) {
                                                                                              const auto edge = graph.findEdge(u,v);
                                                                                              if (edge >=0 ){
                                                                                                  array::StaticArray<int64_t, DIM + 1> affIndex;
                                                                                                  std::copy(std::begin(coordP), std::end(coordP), std::begin(affIndex));
                                                                                                  affIndex[DIM] = i;
                                                                                                  const auto aff_value = affinities[affIndex];
                                                                                                  if (aff_value > maxAff(threadId, edge))
                                                                                                      maxAff(threadId, edge) = aff_value;
                                                                                                  counter(threadId,edge) += affinitiesWeights[affIndex];
                                                                                                  accAff(threadId,edge) += aff_value * affinitiesWeights[affIndex];
                                                                                              }
                                                                                          }
                                                                                      }
                                                                                  }
                                                                              }
                                                                          }
                                                                      });
                          }

                          // Create outputs:
                          typedef xt::pytensor<float, 1> OutArrayType;
                          typedef std::tuple<OutArrayType, OutArrayType, OutArrayType>  OutType;
                          OutArrayType accAff_out = xt::zeros<float>({nb_edges});
                          OutArrayType counter_out = xt::zeros<float>({nb_edges});
                          OutArrayType maxAff_out = xt::zeros<float>({nb_edges});

                          // Normalize:
                          for(auto i=0; i<nb_edges; ++i){
                              maxAff_out(i) = maxAff(0,i);
                              for(auto thr=1; thr<actualNumberOfThreads; ++thr){
                                  counter(0,i) += counter(thr,i);
                                  accAff(0,i) += accAff(thr,i);
                                  if (maxAff(thr,i) > maxAff_out(i))
                                      maxAff_out(i) = maxAff(thr,i);
                              }
                              if(counter(0,i)>0.    ){
                                  accAff_out(i) = accAff(0,i) / counter(0,i);
                                  counter_out(i) = counter(0,i);
                              } else {
                                  accAff_out(i) = 0.;
                                  counter_out(i) = 0.;
                              }
                          }
                          return OutType(accAff_out, maxAff_out, counter_out);;
                      },
                      py::arg("graph"),
                      py::arg("labels").noconvert(),
                      py::arg("affinities").noconvert(),
                      py::arg("affinitiesWeights").noconvert(),
                      py::arg("offsets"),
                      py::arg("hasIgnoreLabel"),
                      py::arg("ignoreLabel"),
                      py::arg("numberOfThreads") = -1
        );
        module.def("accumulateAffinitiesMeanAndLengthInsideClusters_impl_",
                      [](
                              xt::pytensor<LABELS_TYPE, DIM> labels,
                              const LABELS_TYPE maxLabel,
                              xt::pytensor<float, DIM + 1> affinities,
                              xt::pytensor<int, 2> offsets,
                              xt::pytensor<float, 1> offsetWeights,
                              const bool hasIgnoreLabel,
                              const uint64_t ignoreLabel,
                              const int numberOfThreads
                      ) {
                          array::StaticArray<uint64_t, DIM> shape;

                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape()[d];
                              NIFTY_CHECK_OP(shape[d],==,affinities.shape()[d], "affinities have wrong shape");
                          }
                          NIFTY_CHECK_OP(offsets.shape()[0],==,affinities.shape()[DIM], "Affinities and offsets do not match");
                          NIFTY_CHECK_OP(offsets.shape()[0], ==, offsetWeights.shape()[0], "Offsets weights and offsets do not match");

                          // Create thread pool:
                          nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                          nifty::parallel::ThreadPool threadpool(pOpts);
                          const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                          // Create temporary arrays
                          const std::size_t outArray_size = uint64_t(maxLabel + 1);
                          xt::pytensor<float, 2> accAff = xt::zeros<float>({actualNumberOfThreads, outArray_size});
                          xt::pytensor<float, 2> maxAff = xt::zeros<float>({actualNumberOfThreads, outArray_size});
                          xt::pytensor<float, 2> counter = xt::zeros<float>({actualNumberOfThreads, outArray_size});
                          {
                              py::gil_scoped_release allowThreads;
                              nifty::tools::parallelForEachCoordinate(threadpool,
                                                                      shape,
                                                                      [&](const auto threadId, const auto & coordP){
                                                                          const auto u = labels[coordP];
                                                                          if ((!hasIgnoreLabel) || (hasIgnoreLabel && u!=ignoreLabel)) {
                                                                              for(auto i=0; i<offsets.shape()[0]; ++i){
                                                                                  auto coordQ = coordP;
                                                                                  for (auto d=0; d<DIM; ++d) {
                                                                                      coordQ[d] += offsets(i,d);
                                                                                  }
                                                                                  if(coordQ.allInsideShape(shape)){
                                                                                      const auto v = labels[coordQ];
                                                                                      // Check if there is ignore label:
                                                                                      if ((!hasIgnoreLabel) || (hasIgnoreLabel && v!=ignoreLabel)) {
                                                                                          // Only consider if both pixels are inside the same cluster:
                                                                                          if(u == v){
                                                                                              array::StaticArray<int64_t, DIM + 1> affIndex;
                                                                                              std::copy(std::begin(coordP), std::end(coordP), std::begin(affIndex));
                                                                                              affIndex[DIM] = i;
                                                                                              const auto aff_value = affinities[affIndex];
                                                                                              if (aff_value > maxAff(threadId, u))
                                                                                                  maxAff(threadId, u) = aff_value;
                                                                                              counter(threadId,u) += offsetWeights(i);
                                                                                              accAff(threadId,u) += aff_value * offsetWeights(i);
                                                                                          }
                                                                                      }
                                                                                  }
                                                                              }
                                                                          }
                                                                      });
                          }

                          // Create outputs:
                          typedef xt::pytensor<float, 1> OutArrayType;
                          typedef std::tuple<OutArrayType, OutArrayType, OutArrayType>  OutType;
                          OutArrayType accAff_out = xt::zeros<float>({outArray_size});
                          OutArrayType counter_out = xt::zeros<float>({outArray_size});
                          OutArrayType maxAff_out = xt::zeros<float>({outArray_size});

                          // Normalize:
                          for(auto i=0; i<maxLabel+1; ++i){
                              maxAff_out(i) = maxAff(0,i);
                              for(auto thr=1; thr<actualNumberOfThreads; ++thr){
                                  counter(0,i) += counter(thr,i);
                                  accAff(0,i) += accAff(thr,i);
                                  if (maxAff(thr,i) > maxAff_out(i))
                                      maxAff_out(i) = maxAff(thr,i);
                              }
                              if(counter(0,i)>0.){
                                  accAff_out(i) = accAff(0,i) / counter(0,i);
                                  counter_out(i) = counter(0,i);
                              } else {
                                  accAff_out(i) = 0.;
                                  counter_out(i) = 0.;
                              }
                          }
                          return OutType(accAff_out, maxAff_out, counter_out);;
                      },
                      py::arg("labels").noconvert(),
                      py::arg("maxLabel"),
                      py::arg("affinities").noconvert(),
                      py::arg("offsets"),
                      py::arg("offsetWeights"),
                      py::arg("hasIgnoreLabel"),
                      py::arg("ignoreLabel"),
                      py::arg("numberOfThreads") = -1
        );
    }

    void exportAccumulateLongRangeAffinities(py::module & module) {
        typedef PyUndirectedGraph GraphType;
        exportAccumulateAffinitiesMeanAndLength<3, GraphType, uint64_t>(module);
        exportAccumulateAffinitiesMeanAndLength<3, GraphType, uint32_t>(module);
    }
}
}
