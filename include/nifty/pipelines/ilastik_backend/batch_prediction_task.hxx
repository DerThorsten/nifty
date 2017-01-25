#pragma once

#include "nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx"
#include "nifty/pipelines/ilastik_backend/feature_computation_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_task.hxx"

namespace nifty{
namespace pipelines{
namespace ilastik_backend{
            
    template<unsigned DIM>
    class batch_prediction_task : public tbb::task
    {
    private:
        using data_type = float;
        using in_data_type = uint8_t;
        using coordinate = nifty::array::StaticArray<int64_t, DIM>;
        
        using float_array = nifty::marray::Marray<data_type>;
        using float_array_view = nifty::marray::View<data_type>;
        
        using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
        using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
        using raw_cache = Hdf5Input<in_data_type, DIM, false, in_data_type>;
        using random_forest_vector = RandomForestVectorType;

        using blocking = tools::Blocking<DIM>;
        
        // empty constructor doing nothig
        //batch_prediction_task() :
        //{}
        

    public:

        // TODO we want to change the strings here to some simpler flags at some point
        using selection_type = std::pair<std::vector<std::string>,std::vector<double>>;
        
        // construct batch prediction for single input
        batch_prediction_task(const std::string & prediction_file,
                const std::string & prediction_key,
                const std::string & rf_file,
                const std::string & rf_key,
                const selection_type & selected_features,
                const coordinate  & block_shape,
                const size_t max_num_cache_entries) :
            blocking_(),
            rawCache_( hdf5::Hdf5Array<in_data_type>( hdf5::openFile(prediction_file), prediction_key ) ),
            rfFile_(rf_file),
            rfKey_(rf_key),
            featureCache_(),
            predictionCache_(),
            selectedFeatures_(selected_features),
            blockShape_(block_shape),
            maxNumCacheEntries_(max_num_cache_entries),
            rfVectors_() 
        {
            init();
        }

        
        void init() {

            // init the blocking
            coordinate volBegin = coordinate({0,0,0});
            const coordinate & volShape = rawCache_.spaceTimeShape();
            blocking_ = std::make_unique<blocking>(volBegin, volShape, blockShape_);

            // init the feature cache
            std::function<float_array_view(size_t)> retrieve_features_for_caching = [&](size_t blockId) -> float_array_view {
               float_array out_array(blockShape_.begin(), blockShape_.end());
               feature_computation_task<DIM> & feat_task = *new(tbb::task::allocate_child()) feature_computation_task<DIM>(blockId, rawCache_, out_array, selectedFeatures_, *blocking_);
               // TODO why ref-count 2
               set_ref_count(2);
               // TODO spawn or spawn_and_wait
               spawn_and_wait_for_all(feat_task);
               //spawn(feat_task)
               return out_array;
            };
            
            featureCache_ = std::make_unique<feature_cache>(retrieve_features_for_caching, maxNumCacheEntries_);
            
            // TODO use vigra rf 3 instead !
            get_rfs_from_file(rfVectors_, rfFile_, rfKey_, 4);
            
            // init the prediction cache
            std::function<float_array_view(size_t)> retrieve_prediction_for_caching = [&](size_t blockId) -> float_array_view {
               float_array out_array(blockShape_.begin(), blockShape_.end());
               random_forest_prediction_task<DIM> & rf_task = *new(tbb::task::allocate_child()) random_forest_prediction_task<DIM>(blockId, *featureCache_, out_array, rfVectors_);
               // TODO why ref count 2
               set_ref_count(2);
               // TODO spawn or spawn_and_wait
               spawn_and_wait_for_all(rf_task);
               //spawn(rf_task)
               return out_array;
            };

            predictionCache_ = std::make_unique<prediction_cache>(retrieve_prediction_for_caching, maxNumCacheEntries_);
        }
    
    
        tbb::task* execute() {
            // TODO spawn the tasks to batch process the complete volume
            return NULL;
        }


    private:
        // global blocking
        std::unique_ptr<blocking> blocking_;
        raw_cache        rawCache_;
        std::string rfFile_;
        std::string rfKey_;
        std::unique_ptr<feature_cache> featureCache_;
        std::unique_ptr<prediction_cache> predictionCache_;
        selection_type selectedFeatures_;
        coordinate blockShape_;
        size_t maxNumCacheEntries_;
        random_forest_vector rfVectors_;
    };

} // namespace ilastik_backend
} // namepsace pipelines
} // namespace nifty
