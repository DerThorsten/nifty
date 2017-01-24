#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_operator.hxx"

namespace nifty
{
    namespace pipelines
    {
        namespace ilastik_backend
        {
            template<unsigned DIM>
            class random_forest_prediction_task : public tbb::task 
            {
            public:
                // typedefs
                using random_forest_prediction_operator = operatorilastikbackend::operators::random_forest_prediction_operator<DIM>;
                using feature_cache = ???;
                using data_type = float;
                using float_array_view = nifty::marray::View<data_type>;
                using random_forest_vector = nifty::pipelines::ilastik_backend::RandomForestVectorType;
            public:
                // API
                random_forest_prediction_task(
                    size_t blockId,
                    feature_cache& fc,
                    float_array_view& out,
                    random_forest_vector& random_forests
                ):
                    blockId_(blockId),
                    feature_cache_(fc),
                    out_array_(out),
                    random_forest_vector_(random_forests)
                {
                }

                tbb::task* execute()
                {
                    // ask for features. This blocks if it's not present
                    feature_cache::handle_object ho = fc[blockId];
                    float_array_view& features = ho.value();
                    compute(features);
                }

                void compute(const float_array_view & in)
                {
                    // TODO: transform data to vigra array?!
                    size_t pixel_count = 1;
                    for(size_t i = 1; i < DIM; i++)
                    {
                        pixel_count *= in.shape(i);
                    }

                    size_t num_pixel_classification_labels = random_forest_vector_[0].class_count();
                    size_t num_required_features = random_forest_vector_[0].feature_count();
                    assert(num_required_features == in.shape(0));

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
                    &out_array_(0) = vigra_in.data();
                }

            private:
                // members
                size_t blockId_;
                feature_cache& feature_cache_;
                float_array_view& out_array_;
                random_forest_vector& random_forest_vector_;
        }
    }
}

#define _RANDOM_FOREST_PREDICTION_TASK_H_
