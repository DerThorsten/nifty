#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"
#include "nifty/tools/label_multiset.hxx"

namespace py = pybind11;

namespace nifty{
namespace tools{

    template<unsigned NDIM, class ID_TYPE, class COUNT_TYPE>
    void exportLabelMultisetT(py::module & m) {

        typedef ID_TYPE IdType;
        typedef COUNT_TYPE CountType;
        typedef std::size_t OffsetType;

        typedef xt::pytensor<OffsetType, 1> OffsetVector;
        typedef xt::pytensor<IdType, 1> IdVector;
        typedef xt::pytensor<CountType, 1> CountVector;

        m.def("readSubset", [](const OffsetVector & offsets,
                               const OffsetVector & sizes,
                               const IdVector & ids,
                               const CountVector & counts,
                               const bool argsort){
            std::vector<IdType> ids_tmp;
            std::vector<CountType> counts_tmp;
            {
                py::gil_scoped_release lift_gil;
                readSubset(offsets, sizes, ids, counts, ids_tmp, counts_tmp, argsort);
            }

            // TODO can we use xt::adapt here instead of copying values?
            const int64_t n_ids = ids_tmp.size();
            IdVector ids_out = xt::zeros<IdType>({n_ids});
            CountVector counts_out = xt::zeros<CountType>({n_ids});
            {
                py::gil_scoped_release lift_gil;
                for(std::size_t i = 0; i < ids_tmp.size(); ++i) {
                    ids_out(i) = ids_tmp[i];
                    counts_out(i) = counts_tmp[i];
                }
            }
            return std::make_pair(ids_out, counts_out);
        }, py::arg("offsets"), py::arg("sizes"), py::arg("ids"), py::arg("counts"),
           py::arg("argsort")=true);


        m.def("downsampleMultiset", [](const Blocking<NDIM> & blocking,
                                       const OffsetVector & offsets,
                                       const OffsetVector & entry_sizes,
                                       const OffsetVector & entry_offsets,
                                       const IdVector & ids,
                                       const CountVector & counts,
                                       const int restrict_set){
            // argmax and offsets: we know the size already and can allocate the pyarrays
            const int64_t n_blocks = blocking.numberOfBlocks();
            IdVector new_argmax = xt::zeros<IdType>({n_blocks});
            OffsetVector new_offsets = xt::zeros<OffsetType>({n_blocks});

            // ids and counts: we don't know the sizes a priori, so we need vectors
            std::vector<IdType> new_ids;
            std::vector<CountType> new_counts;
            {
                py::gil_scoped_release lift_gil;
                downsampleMultiset(blocking,
                                   offsets, entry_sizes, entry_offsets,
                                   ids, counts, restrict_set,
                                   new_argmax, new_offsets, new_ids, new_counts);
            }

            // TODO can we use xt::adapt here instead of copying values?
            const int64_t n_ids = new_ids.size();
            IdVector ids_out = xt::zeros<IdType>({n_ids});
            CountVector counts_out = xt::zeros<CountType>({n_ids});
            {
                py::gil_scoped_release lift_gil;
                for(std::size_t i = 0; i < new_ids.size(); ++i) {
                    ids_out(i) = new_ids[i];
                    counts_out(i) = new_counts[i];
                }
            }
            return std::make_tuple(new_argmax, new_offsets, ids_out, counts_out);
        }, py::arg("blocking"),
           py::arg("offsets"), py::arg("entry_sizes"), py::arg("entry_offsets"),
           py::arg("ids"), py::arg("counts"), py::arg("restrict_set"));


        typedef MultisetMerger<OffsetType, IdType, CountType> Merger;
        py::class_<Merger>(m, "MultisetMerger")
            .def(py::init<const OffsetVector &,
                          const OffsetVector &,
                          const IdVector &,
                          const CountVector &>(),
                 py::arg("offsets"), py::arg("entry_sizes"),
                 py::arg("ids"), py::arg("counts"))


            .def("get_ids", [](const Merger & self){
                const auto & ids = self.get_ids();
                IdVector ids_out = xt::zeros<IdType>({ids.size()});
                {
                    py::gil_scoped_release lift_gil;
                    for(std::size_t ii = 0; ii < ids.size(); ++ii) {
                        ids_out(ii) = ids[ii];
                    }
                }
                return ids_out;
            })
            .def("get_counts", [](const Merger & self){
                const auto & counts = self.get_counts();
                IdVector counts_out = xt::zeros<IdType>({counts.size()});
                {
                    py::gil_scoped_release lift_gil;
                    for(std::size_t ii = 0; ii < counts.size(); ++ii) {
                        counts_out(ii) = counts[ii];
                    }
                }
                return counts_out;
            })

            .def("update",[](Merger & self,
                             const OffsetVector & unique_offsets,
                             const OffsetVector & entry_sizes,
                             const IdVector & ids,
                             const CountVector & counts,
                             OffsetVector & offsets){
                {
                    py::gil_scoped_release lift_gil;
                    self.update(unique_offsets, entry_sizes, ids, counts, offsets);
                }
                return offsets;
            }, py::arg("unique_offsets"), py::arg("entry_sizes"),
               py::arg("ids"), py::arg("counts"), py::arg("offsets"))
        ;

    }


    void exportLabelMultiset(py::module & m) {
        exportLabelMultisetT<3, uint64_t, int32_t>(m);
    }

}
}
