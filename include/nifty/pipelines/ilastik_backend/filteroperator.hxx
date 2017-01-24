#ifndef _OPERATORS_FILTEROPERATOR_H_
#define _OPERATORS_FILTEROPERATOR_H_

#include <tuple>

#include <tbb/flow_graph.h>

#include <ilastik-backend/operatos/baseoperator.h>
#include <ilastik-backend/types.h>

#include "nifty/marray/marray.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"

// TODO need to think how we handle non float input to the filters
// probably overload the filter operator for uint8
using float_type = flowgraph::job_data<nifty::marray::View<float>>;
// TODO dtype handle at the level of marrays / hdf5 loading from data!
// TODO or maybe move the data copy to the feature task, because we can ship around
// more data with smaller uint8 around -> make all this tomorrow
//using uint8_type = flowgraph::job_data<uint8_t>;

namespace ilastikbackend
{
    namespace operators
    {

        template<unsigned DIM>
        class filter_operator : public base_operator<tuple<float_type>,tuple<float_type>>
        {
        public:
            // definition of enums to use for the slots
            enum class input_slots {
                RAW = 0
            };

            enum class output_slots {
                FEATURES = 0
            };
        private:
            using apply_type = nifty::features::ApplyFilters<DIM>;
            using filter_type = nifty::features::FilterBase;

        public:
            filter_operator(const types::set_of_cancelled_job_ids& setOfCancelledJobIds,
                    const std::vector<std::string> & filter_names,
                    const std::vector<double> & sigma_values,
                    const double outer_scale = 0. ): apply_(sigmas, filter_names)// TODO need to rethink if we want to apply different outer scales for the structure tensor eigenvalues
                base_operator<std::tuple<float_type>, std::tuple<float_type> >(setOfCancelledJobIds)
            {
            }

            virtual std::tuple<float_type> executeImpl(const std::tuple<float_type> & in) const
            {
                // TODO copy input data from uint8 to float here ?!
                // TODO set the window ration thing according to the feature type and halo here or in the constructir
                // TODO is the axis order (channel, space) optimal here ?
                auto & in_data = std::get<input_slots.RAW>(in);
                // allocate the out data
                size_t out_shape[DIM+1];
                out_shape[0] = _apply.numberOfChannels();
                for(int d = 0; d < DIM; ++d)
                    out_shape[d+1] = in_data.shape[d];
                nifty::marray::Marray<float> out_data(out_shape, out_shape+DIM+1);
                // apply the filter via the functor
                _apply(in_data, out_data);

                return std::make_tuple(float_type(in_data.job_id, out_data));
            }

        private:
            apply_type apply_; // the functor for applying the filters
        }
    } // namespace operator
} // namespace ilastik_backend


#endif // _OPERATORS_FILTEROPERATOR_H_
