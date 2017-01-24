#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

//#ifndef TBB_PREVIEW_CONCURRENT_LRU_CACHE
//#define TBB_PREVIEW_CONCURRENT_LRU_CACHE
//#include <tbb/concurrent_lru_cache.h>

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include <nifty/marray/marray.hxx>
// TODO include appropriate vigra stuff
#include <vigra/random_forest_hdf5_impex.hxx>

class tbb::concurrent_lru_cache;

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
                using data_type = float;
                using float_array_view = nifty::marray::View<data_type>;
                using random_forest_vector = nifty::pipelines::ilastik_backend::RandomForestVectorType;
                using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view>;
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
                    feature_cache::handle_object ho = feature_cache_[blockId_];
                    float_array_view& features = ho.value();
                    compute(features);
                    return NULL;
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

                    // copy data from marray to vigra. TODO: is the axes order correct??
                    vigra::MultiArrayView<DIM, data_type> vigra_in(pixel_count * num_required_features, &in(0));

                    vigra::MultiArray<2, data_type> prediction_map_view(vigra::Shape2(pixel_count, num_pixel_classification_labels));

                    // loop over all random forests for prediction probabilities
                    std::cout << "\tPredict RFs" << std::endl;
                    for(size_t rf = 0; rf < random_forest_vector_.size(); rf++)
                    {
                        vigra::MultiArray<2, data_type> prediction_temp(pixel_count, num_pixel_classification_labels);
                        random_forest_vector_[rf].predictProbabilities(vigra_in, prediction_temp);
                        prediction_map_view += prediction_temp;
                    }

                    // divide probs by num random forests
                    prediction_map_view /= random_forest_vector_.size();

                    // transform back to marray 
                    &out_array_(0) = prediction_map_view.data();
                }

            private:
                // members
                size_t blockId_;
                feature_cache& feature_cache_;
                float_array_view& out_array_;
                random_forest_vector& random_forest_vector_;
        };
    }
}

#endif // _RANDOM_FOREST_PREDICTION_TASK_H_
