#ifndef _OPERATORS_FILTEROPERATOR_H_
#define _OPERATORS_FILTEROPERATOR_H_

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE
#include <tbb/concurrent_lru_cache.h>

#include <tuple>

#include <tbb/flow_graph.h>

#include <ilastik-backend/operatos/baseoperator.h>
#include <ilastik-backend/types.h>

#include "nifty/marray/marray.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty {
    namespace ilastikbackend
    {
        namespace operators
        {
    
            template<unsigned DIM>
            class feature_computation_task : public tbb::task
            {
            private:
                using apply_type = nifty::features::ApplyFilters<DIM>;
                using raw_cache = ???;
                using in_data_type = uint8_t;
                using out_data_type = float;
                using out_array_view = nifty::marray::View<out_data_type>;
                using in_array_view = nifty::marray::View<in_data_type>;
                using selected_feature_type = std::pair<std::vector<std::string>,
                      std::vector<double>>;
                using array::StaticArray<int64_t,DIM> coordinate;
    
            public:
                feature_computation_task(,
                        size_t block_id,
                        raw_cache & rc,
                        out_array_view & out,
                        const selected_feature_type & selected_features) :
                    blockId_(block_id),
                    rawCache_(rc),
                    outArray_(out),
                    apply_(selected_features.second, selected_features.first)// TODO need to rethink if we want to apply different outer scales for the structure tensor eigenvalues
                {
                }
                
                tbb::task* execute()
                {
                    // ask for the raw data
                    // TODO get the proper halo and pass it to the cache !!
                    raw_cache::handle_object ho = rc[blockId];
                    in_array_view& data = ho.value();
                    compute(data);
                    return data;
                }
    
                void compute(const in_data_type & in)
                {
                    // TODO set the correct window ratios
                    //copy input data from uint8 to float
                    coordinate in_shape;
                    for(int d = 0; d < DIM; ++d)
                        in_shape[d] = in_data.shape(d);
                    marray::Marray<in_data_type> in_float(in_shape.begin(), in_shape.end());
                    tools::forEachCoordinate(in_shape, [&](const coordinate & coord){
                        in_float(coord.asStdArray()) = in(coord.asStdArray());    
                    });
                    // apply the filter via the functor
                    // TODO consider passing the tbb threadpool here
                    _apply(in_float, outArray_);
                }
    
            private:
                size_t blockId_;
                raw_cache & rawCache_;
                out_array_view & outArray_;
                apply_type apply_; // the functor for applying the filters
            }
        } // namespace operator
    } // namespace ilastik_backend
} // namespace nifty


#endif // _OPERATORS_FILTEROPERATOR_H_
