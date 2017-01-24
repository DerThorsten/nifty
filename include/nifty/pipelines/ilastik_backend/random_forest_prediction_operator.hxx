#ifndef _RANDOM_FOREST_PREDICTION_OPERATOR_H_
#define _RANDOM_FOREST_PREDICTION_OPERATOR_H_

#include <tuple>
#include <assert.h>
#include <tbb/task.h>

#include <vigra/multi_array.hxx>

#include "nifty/marray/marray.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"

// TODO need to think how we handle non float input to the filters
// probably overload the filter operator for uint8
using data_type = float;
using float_array_view = flowgraph::job_data<nifty::marray::View<data_type>>;
// TODO dtype handle at the level of marrays / hdf5 loading from data!
//using uint8_type = flowgraph::job_data<uint8_t>;

namespace ilastikbackend
{
    namespace operators
    {

        template<unsigned DIM>
        class random_forest_prediction_operator
        {
        public:
            random_forest_prediction_operator(
                const nifty::pipelines::ilastik_backend::RandomForestVectorType& random_forest_vector
            ): 
                random_forest_vector_(random_forest_vector)
            {
                assert(random_forest_vector_.size() > 0);
            }

            float_array_view executeImpl(const float_array_view & in) const
            {
                // TODO: transform data to vigra array?!
                size_t pixel_count = 1;
                for(size_t i = 1; i < DIM; i++)
                {
                    pixel_count *= in.shape(i);
                }

                size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
                size_t num_required_features = random_forest_vector_[0].feature_count();
                assert(num_required_features == in.shape(0);

                vigra::MultiArrayView<DIM, data_type> vigra_in(pixel_count * num_required_features, &in(0));

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

                // divide probs by num random forests
                prediction_map_view /= random_forests_.size();

                // transform back to marray 
                float_array_view out(vigra_in.shape());
                &out(0) = vigra_in.data();

                return out;
            }

        private:
            const nifty::pipelines::ilastik_backend::RandomForestVectorType& random_forest_vector_;
        };
    } // namespace operators
} // namespace ilastik_backend


#endif // _RANDOM_FOREST_PREDICTION_OPERATOR_H_
