#include "nifty/pipelines/ilastik_backend/batch_prediction_task.hxx"


int main()
{
    using namespace nifty::pipelines::ilastik_backend;

    constexpr size_t dim  = 3;
    constexpr size_t max_num_entries = 100;
    
    using coordinate = nifty::array::StaticArray<int64_t,dim>;

    // input file
    const std::string in_file = "fake.h5";
    const std::string in_key  = "data";

    // random forests
    const std::string rf_file = "myfile.h5";
    const std::string rf_key = "/somewhere";

    auto feature_selection = std::make_pair( std::vector<std::string>({"GaussianSmoothing"}), std::vector<double>( {1.,2.}) );

    coordinate block_shape({64,64,64}); 

    batch_prediction_task<dim> batch(
            in_file, in_key,
            rf_file, rf_key,
            feature_selection,
            block_shape, max_num_entries);
}
