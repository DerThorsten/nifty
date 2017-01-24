#define TBB_PREVIEW_CONCURRENT_LRU_CACHE
#include <tbb/concurrent_lru_cache.h>
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"

using namespace nifty::pipelines::ilastik_backend;

int main()
{
    using data_type = float;
    using float_array = nifty::marray::Marray<data_type>;
    using float_array_view = nifty::marray::View<data_type>;
    using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view>;
    using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view>;
    constexpr size_t max_num_entries = 100;
    constexpr std::vector<size_t> blockSize = {64, 64, 64};


    // load random forests
    const std::string rf_filename = "myfile.h5";
    const std::string rf_path = "/somewhere";
    RandomForestVectorType rf_vector;
    get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    MySuperDuperHDF5Cache rc;

    feature_cache fc([&rc, &selected_features](size_t blockId) -> float_array_view {
        float_array out_array(blockSize.begin(), blockSize.end());
        feature_computation_task& rf_task = *tbb::new(tbb::allocate_child()) feature_computation_task(blockId, rc, out_array, selected_features);
        set_ref_count(2);
        // Start a running and wait for all children (a and b).
        spawn_and_wait_for_all(rf_task);
        return out_array;
    }, max_num_entries);

    prediction_cache pc([&fc, &rf_vector](size_t blockId) -> float_array_view {
        float_array out_array(blockSize.begin(), blockSize.end());
        random_forest_prediction_task& rf_task = *tbb::new(tbb::allocate_child()) random_forest_prediction_task(blockId, fc, out_array, rf_vector);
        set_ref_count(2);
        // Start a running and wait for all children (a and b).
        spawn_and_wait_for_all(rf_task);
        return out_array;
    }, max_num_entries);

    handle = pc[15];
}