#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <tbb/concurrent_lru_cache.h>

#include "nifty/pipelines/ilastik_backend/feature_computation_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include "nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx"

#include "nifty/tools/blocking.hxx"


int main()
{
    using namespace nifty::pipelines::ilastik_backend;
    
    constexpr size_t dim  = 3;

    using data_type = float;
    using in_data_type = uint8_t;
    using float_array = nifty::marray::Marray<data_type>;
    using float_array_view = nifty::marray::View<data_type>;
    using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    constexpr size_t max_num_entries = 100;
    using raw_cache = Hdf5Input<in_data_type, dim, false, in_data_type>;
    using coordinate = nifty::array::StaticArray<int64_t, dim>;


    // load random forests
    const std::string rf_filename = "myfile.h5";
    const std::string rf_path = "/somewhere";
    RandomForestVectorType rf_vector;
    get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    std::string raw_file = "./fake.h5";
    auto file = nifty::hdf5::openFile(raw_file);
    nifty::hdf5::Hdf5Array<in_data_type> raw_array(file, "data");
    raw_cache rc(raw_array);
    coordinate fake_shape({512,512,512});
    coordinate begin({0,0,0});
    coordinate blockShape({64,64,64});
    nifty::tools::Blocking<dim> blocking(begin, fake_shape, blockShape);

    auto selected_features = std::make_pair(std::vector<std::string>({"GaussianSmoothing"}),
            std::vector<double>({2.,3.}));

    std::function<float_array_view(size_t)> retrieve_features_for_caching = [&rc, &selected_features, &blockShape, &blocking, dim](size_t blockId) -> float_array_view {
       float_array out_array(blockShape.begin(), blockShape.end());
       // FIXME allocate_child must be called with obj
       // feature_computation_task& feat_task = *new(tbb::task::allocate_child()) feature_computation_task(blockId, rc, out_array, selected_features, blocking);
       feature_computation_task<dim> & feat_task = *new(tbb::task::allocate_root()) feature_computation_task<dim>(blockId, rc, out_array, selected_features, blocking);
       feat_task.set_ref_count(2);
       // FIXME this can't be invoked w/o object
       //tbb::task::spawn_and_wait_for_all(feat_task);
       tbb::task::spawn_root_and_wait(feat_task);
       return out_array;
    };

    feature_cache fc(retrieve_features_for_caching, max_num_entries);

    std::function<float_array_view(size_t)> retrieve_prediction_for_caching = [&fc, &blockShape, &rf_vector, dim](size_t blockId) -> float_array_view {
       float_array out_array(blockShape.begin(), blockShape.end());
       random_forest_prediction_task<dim> & rf_task = *new(tbb::task::allocate_root()) random_forest_prediction_task<dim>(blockId, fc, out_array, rf_vector);
       //random_forest_prediction_task& rf_task = *tbb::new(tbb::allocate_child()) random_forest_prediction_task(blockId, fc, out_array, rf_vector);
       rf_task.set_ref_count(2);
       // Start a running and wait for all children (a and b).
       //spawn_and_wait_for_all(rf_task);
       tbb::task::spawn_root_and_wait(rf_task);
       return out_array;
    };

    prediction_cache pc(retrieve_prediction_for_caching, max_num_entries);

    // handle = pc[15];
    
    return 0;
}
