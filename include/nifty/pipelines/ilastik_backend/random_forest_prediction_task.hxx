#ifndef _RANDOM_FOREST_PREDICTION_TASK_H_
#define _RANDOM_FOREST_PREDICTION_TASK_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include <tbb/tbb.h>
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include <nifty/marray/marray.hxx>
// TODO include appropriate vigra stuff
#include <vigra/random_forest_hdf5_impex.hxx>

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
                using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
                using out_shape_type = array::StaticArray<int64_t,DIM+1>;

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
                    feature_cache::handle ho = feature_cache_[blockId_];
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
                    vigra::MultiArrayView<2, data_type> vigra_in(vigra::Shape2(pixel_count, num_required_features), &in(0));

                    vigra::MultiArray<2, data_type> prediction_map_view(vigra::Shape2(pixel_count, num_pixel_classification_labels));

                    // loop over all random forests for prediction probabilities
                    std::cout << "\tPredict RFs" << std::endl;
                    for(size_t rf = 0; rf < random_forest_vector_.size(); ++rf)
                    {
                        vigra::MultiArray<2, data_type> prediction_temp(pixel_count, num_pixel_classification_labels);
                        random_forest_vector_[rf].predictProbabilities(vigra_in, prediction_temp);
                        prediction_map_view += prediction_temp;
                    }

                    // divide probs by num random forests
                    prediction_map_view /= random_forest_vector_.size();

                    // transform back to marray
                    out_shape_type output_shape;
                    for(size_t d = 0; d < DIM; ++d) {
                        output_shape[d] = out_array_.shape(0);
                    }
                    output_shape[DIM] = num_required_features;
                    float_array_view& tmp_out_array = out_array_;
                    tools::forEachCoordinate(output_shape, [&tmp_out_array, &prediction_map_view, output_shape](const out_shape_type& coord)
                    {
                        size_t pixelRow = coord[0] * (output_shape[1] + output_shape[2]) + coord[1] * (output_shape[2]);
                        if(DIM == 3)
                        {
                            pixelRow = coord[0] * (output_shape[1] + output_shape[2] + output_shape[3]) + coord[1] * (output_shape[2] + output_shape[3]) + coord[2] * output_shape[3];
                        }
                        tmp_out_array(coord.asStdArray()) = prediction_map_view(pixelRow, coord[DIM]);
                    });
                    
                }

            private:
                // members
                size_t blockId_;
                feature_cache& feature_cache_;
                float_array_view& out_array_;
                random_forest_vector& random_forest_vector_;
            };
        
        } // namespace ilastik_backend
    } // namespace pipelines
} // namespace nifty

#endif // _RANDOM_FOREST_PREDICTION_TASK_H_
