#ifndef _RANDOM_FOREST_PREDICTION_OPERATOR_H_
#define _RANDOM_FOREST_PREDICTION_OPERATOR_H_

#include <tuple>
#include <assert.h>
#include <tbb/flow_graph.h>

#include <ilastik-backend/operatos/baseoperator.h>
#include <ilastik-backend/types.h>

#include "nifty/marray/marray.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"

// TODO need to think how we handle non float input to the filters
// probably overload the filter operator for uint8
using data_type = float;
using float_type = flowgraph::job_data<nifty::marray::View<data_type>>;
// TODO dtype handle at the level of marrays / hdf5 loading from data!
//using uint8_type = flowgraph::job_data<uint8_t>;

namespace ilastikbackend
{
    namespace operators
    {

        template<unsigned DIM>
        class random_forest_prediction_operator : public base_operator<std::tuple<float_type>, std::tuple<float_type>> // TODO how should we inherit? I guess private ?!
        {
        public:
            // definition of enums to use for the slots
            enum class input_slots {
                FEATURES = 0
            };

            enum class output_slots {
                PROBABILITIES = 0
            };

        public:
            random_forest_prediction_operator(
                const types::set_of_cancelled_job_ids& sef_of_cancelled_job_ids,
                const nifty::pipelines::ilastik_backend::RandomForestVectorType& random_forest_vector
            ): 
                base_operator<std::tuple<float_type>, std::tuple<float_type> >(sef_of_cancelled_job_ids),
                random_forest_vector_(random_forest_vector)
            {
                assert(random_forest_vector_.size() > 0);
            }

            virtual std::tuple<float_type> executeImpl(const std::tuple<float_type> & in) const
            {
                auto & in_data = std::get<input_slots.FEATURES>(in);
                // TODO: transform data to vigra array?!
                size_t pixel_count = in_data.dim(0) * in_data.dim(1);
                size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
                size_t num_required_features = random_forest_vector_[0].feature_count();
                assert(num_required_features == in_data.featureDims());

                vigra::MultiArrayView<2, data_type> prediction_map_view(
                    vigra::Shape2(pixel_count, num_pixel_classification_labels),
                    segmentation.prediction_map_.data());
                // loop over all random forests for prediction probabilities
                std::cout << "\tPredict RFs" << std::endl;
                for(size_t rf = 0; rf < random_forests_.size(); rf++)
                {
                    vigra::MultiArray<2, data_type> prediction_temp(pixel_count, num_pixel_classification_labels);
                    random_forests_[rf].predictProbabilities(feature_view, prediction_temp);
                    prediction_map_view += prediction_temp;
                }

                // TODO: transform back to marray
                // TODO: divide probs by num random forests
                // TODO: return 1-element tuple with resulting probabilities
            }

        private:
            const nifty::pipelines::ilastik_backend::RandomForestVectorType& random_forest_vector_;
        }
    } // namespace operators
} // namespace ilastik_backend


#endif // _RANDOM_FOREST_PREDICTION_OPERATOR_H_
