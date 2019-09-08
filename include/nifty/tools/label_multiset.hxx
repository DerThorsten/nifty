#include <unordered_map>
#include <boost/functional/hash.hpp>
#include "xtensor/xtensor.hpp"
#include "nifty/tools/blocking.hxx"

namespace nifty {
namespace tools {


    // this stack-overflow offers some ideas how to do this without copies, but
    // for now this would be premature:
    // https://stackoverflow.com/questions/838384/reorder-vector-using-a-vector-of-indices
    template<class V>
    inline void reorder_inplace(V & v, const std::vector<std::size_t > & idx) {
        V tmp(v.size());
        for(std::size_t i = 0; i < v.size(); ++i) {
            tmp[i] = v[idx[i]];
        }
        v.swap(tmp);
    }


    // inplace argsort of both vectors by the first vector
    // NOTE we assume both have the same size !
    template<class V1, class V2>
    inline void argsort_by_first_vector(V1 & v1, V2 & v2) {
        std::vector<std::size_t> idx(v1.size());
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
                 [&v1](std::size_t i1, std::size_t i2) {return v1[i1] < v1[i2];});

        // reorder both vectors inplace
        reorder_inplace(v1, idx);
        reorder_inplace(v2, idx);
    }


    template<class OFFSETS, class IDS, class COUNTS>
    inline void readSubset(const OFFSETS & offsets,
                           const OFFSETS & sizes,
                           const IDS & ids,
                           const COUNTS & counts,
                           std::vector<typename IDS::value_type> & ids_out,
                           std::vector<typename COUNTS::value_type> & counts_out,
                           const bool argsort){

        typedef typename IDS::value_type IdType;
        typedef typename COUNTS::value_type CountType;
        typedef std::unordered_map<IdType, CountType> DictType;
        DictType count_dict;

        const std::size_t n_offsets = offsets.size();
        for(std::size_t off_id = 0; off_id < n_offsets; ++off_id) {
            const std::size_t offset = offsets[off_id];
            const std::size_t size = sizes[off_id];

            for(std::size_t pos = offset; pos < offset + size; ++pos) {
                const IdType id = ids(pos);
                const CountType count = counts(pos);

                auto c_it = count_dict.find(id);
                if(c_it == count_dict.end()) {
                    count_dict.emplace(id, count);
                } else {
                    c_it->second += count;
                }
            }
        }

        // copy to the output vectors
        const std::size_t size = count_dict.size();
        ids_out.resize(size);
        counts_out.resize(size);
        std::size_t i = 0;
        for(const auto & elem : count_dict) {
            ids_out[i] = elem.first;
            counts_out[i] = elem.second;
            ++i;
        }

        // argsort the vectors by the id if specified
        if(argsort) {
            argsort_by_first_vector(ids_out, counts_out);
        }
    }


    template<class BLOCK, class STRIDES, class OFFSETS, class IDS, class COUNTS>
    inline void readSubset(const BLOCK & block,
                           const STRIDES & strides,
                           const OFFSETS & offsets,
                           const OFFSETS & entry_sizes,
                           const OFFSETS & entry_offsets,
                           const IDS & ids,
                           const COUNTS & counts,
                           std::vector<typename IDS::value_type> & ids_out,
                           std::vector<typename COUNTS::value_type> & counts_out,
                           const bool argsort=true) {
        typedef typename IDS::value_type IdType;
        typedef typename COUNTS::value_type CountType;

        std::vector<std::size_t> this_offsets, this_sizes;
        const auto & block_begin = block.begin();
        const auto & block_end = block.end();

        // TODO make nd with imlib trick
        // iterate over points in the block and get the entry offsets and sizes
        for(std::size_t x = block_begin[0]; x < block_end[0]; ++x) {
            for(std::size_t y = block_begin[1]; y < block_end[1]; ++y) {
                for(std::size_t z = block_begin[2]; z < block_end[2]; ++z) {
                    const std::size_t index = strides[0] * x + strides[1] * y + strides[2] * z;
                    this_offsets.emplace_back(offsets(index));
                    this_sizes.emplace_back(entry_sizes(entry_offsets(index)));
                }
            }
        }

        // read the subset
        readSubset(this_offsets, this_sizes, ids, counts, ids_out, counts_out, argsort);
    }


    template<class QIT, class IT>
    inline bool check_range(QIT qbegin, QIT qend, IT begin, IT end){
        // check if sizes agree
        if(std::distance(begin, end) != std::distance(qbegin, qend)) {
            return false;
        }

        // check if ids agree
        if(!std::equal(begin, end, qbegin)) {
            return false;
        }

        return true;
    }


    template<class BLOCKING, class OFFSETS, class IDS, class COUNTS>
    inline void downsampleMultiset(const BLOCKING & blocking,
                                   const OFFSETS & offsets,
                                   const OFFSETS & entry_sizes,
                                   const OFFSETS & entry_offsets,
                                   const IDS & ids,
                                   const COUNTS & counts,
                                   const int restrict_set,
                                   IDS & new_argmax,
                                   OFFSETS & new_offsets,
                                   std::vector<typename IDS::value_type> & new_ids,
                                   std::vector<typename COUNTS::value_type> & new_counts) {
        typedef typename IDS::value_type IdType;
        typedef typename COUNTS::value_type CountType;

        typedef std::pair<IdType, CountType> HashKey;
        typedef boost::hash<HashKey> Hash;
        typedef std::vector<std::size_t> HashElement;
        typedef std::unordered_map<HashKey, HashElement, Hash> HashDictType;
        HashDictType candidate_dict;

        const std::size_t n_blocks = blocking.numberOfBlocks();
        // NOTE we assume roiBegin is 1 here, would be good to generalize this
        const auto & shape = blocking.roiEnd();
        std::vector<std::size_t> strides(shape.size());
        strides[strides.size() - 1] = 1;
        for(int d = strides.size() - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        std::vector<std::size_t> new_entry_offsets;
        std::vector<std::size_t> new_entry_sizes;

        std::size_t current_candidate_id = 0;
        for(std::size_t block_id = 0; block_id < n_blocks; ++block_id) {
            // 1.) read the subset from this block
            std::vector<IdType> this_ids;
            std::vector<CountType> this_counts;
            const auto block = blocking.getBlock(block_id);
            readSubset(block, strides,
                       offsets, entry_sizes, entry_offsets,
                       ids, counts, this_ids, this_counts);

            // 2.) apply restrict sets if specified; compute argmax label and count
            IdType max_label;
            CountType max_count;
            if(restrict_set > 0 && this_ids.size() > restrict_set) {
                // arg-sort by counts
                // could use std::nth_element to index sort and only get the 'restrict_set' largest
                // elements, but that's premature optimization for now, because it will complicate the code
                // quite a bit
                argsort_by_first_vector(this_counts, this_ids);
                max_label = this_ids[0];
                max_count = this_counts[0];
                // restrict
                this_ids.resize(restrict_set);
                this_counts.resize(restrict_set);
                // argsort by ids
                argsort_by_first_vector(this_ids, this_counts);
            } else {
                auto max_it = std::max_element(this_counts.begin(), this_counts.end());
                max_label = this_ids[std::distance(this_counts.begin(), max_it)];
                max_count = *max_it;
            }
            new_argmax(block_id) = max_label;

            // 3.) check if we have this entry already in the hashed candidates
            HashKey hash(max_label, max_count);
            bool add_entry = true;
            auto candidate_it = candidate_dict.find(hash);
            if(candidate_it != candidate_dict.end()) {
                const auto & candidates = candidate_it->second;
                // iterate over the candidate offsets
                for(const std::size_t c_offset_id : candidates) {
                    // get the entry offset and entry size for this candidate
                    const std::size_t c_offset = new_entry_offsets[c_offset_id];
                    const std::size_t c_size = new_entry_sizes[c_offset_id];

                    // check the ids
                    bool match = check_range(this_ids.begin(), this_ids.end(),
                                             new_ids.begin() + c_offset,
                                             new_ids.begin() + c_offset + c_size);
                    // if the ids match, check the counts
                    if(match) {
                        match = check_range(this_counts.begin(), this_counts.end(),
                                            new_counts.begin() + c_offset,
                                            new_counts.begin() + c_offset + c_size);
                    }

                    // the candidates and this entry agree -> we skip making a new
                    // entry and just push back the offset we found
                    if(match) {
                        new_offsets(block_id) = c_offset;
                        add_entry = false;
                        break;
                    }
                }
            }

            // 4.) if we haven't found this entry, add it!
            if(add_entry) {
                // update the new_offsets, entry_offsets and entry sizes
                const std::size_t this_offset = new_ids.size();
                new_offsets(block_id) = this_offset;
                new_entry_offsets.emplace_back(this_offset);
                new_entry_sizes.emplace_back(this_ids.size());

                // store ids and counts
                new_ids.insert(new_ids.end(), this_ids.begin(), this_ids.end());
                new_counts.insert(new_counts.end(), this_counts.begin(), this_counts.end());

                // add this offset to the candidates
                if(candidate_it == candidate_dict.end()) {
                    candidate_dict.emplace(hash, std::vector<std::size_t>({current_candidate_id}));
                } else {
                    candidate_it->second.emplace_back(current_candidate_id);
                }

                // increase the current candidate id
                ++current_candidate_id;
            }
        }
    }


    template<class OFFSET_TYPE, class ID_TYPE, class COUNT_TYPE>
    class MultisetMerger {
    public:
        typedef OFFSET_TYPE OffsetType;
        typedef ID_TYPE IdType;
        typedef COUNT_TYPE CountType;

        typedef std::pair<IdType, CountType> HashKey;
        typedef boost::hash<HashKey> Hash;
        typedef std::vector<std::size_t> HashElement;
        typedef std::unordered_map<HashKey, HashElement, Hash> HashDictType;

        template<class OFFSETS, class IDS, class COUNTS>
        MultisetMerger(const OFFSETS & offsets,
                       const OFFSETS & entry_sizes,
                       const IDS & ids,
                       const COUNTS & counts) : offsets_(offsets.begin(), offsets.end()),
                                                entry_sizes_(entry_sizes.begin(), entry_sizes.end()),
                                                ids_(ids.begin(), ids.end()),
                                                counts_(counts.begin(), counts.end()){
            init_hashed();
        }

        inline const std::vector<IdType> & get_ids() const {
            return ids_;
        }
        inline const std::vector<CountType> & get_counts() const {
            return counts_;
        }


        template<class OFFSETS, class IDS, class COUNTS>
        inline void update(const OFFSETS & unique_offsets,
                           const OFFSETS & entry_sizes,
                           const IDS & ids,
                           const COUNTS & counts,
                           OFFSETS & offsets){

            const std::size_t n_entries = unique_offsets.size();
            // this dict maps entry id (w.r.t. inputs!) to absolute element offset
            std::unordered_map<OffsetType, OffsetType> new_offset_dict;

            // iterate over entries and check if we have them hashed already
            for(std::size_t entry = 0; entry < n_entries; ++entry) {
                const OffsetType off = unique_offsets[entry];
                const OffsetType size = entry_sizes[entry];

                auto ids_begin = ids.begin() + off;
                auto ids_end = ids.begin() + off + size;
                auto counts_begin = counts.begin() + off;
                auto counts_end = counts.begin() + off + size;

                auto max_it = std::max_element(counts_begin, counts_end);
                const IdType max_label = *(ids_begin + std::distance(counts_begin, max_it));
                const CountType max_count = *max_it;

                HashKey hash(max_label, max_count);
                auto hash_it = hashed_.find(hash);

                bool new_entry = true;
                if(hash_it != hashed_.end()) {

                    // iterate over the candidate offsets
                    const auto & candidates = hash_it->second;
                    for(const std::size_t c_offset_id : candidates) {

                        // get the entry offset and entry size for this candidate
                        const std::size_t c_offset = offsets_[c_offset_id];
                        const std::size_t c_size = entry_sizes_[c_offset_id];

                        // check the ids
                        bool match = check_range(ids_begin, ids_end,
                                                 ids_.begin() + c_offset,
                                                 ids_.begin() + c_offset + c_size);

                        // if the ids match, check the counts
                        if(match) {
                            bool match = check_range(counts_begin, counts_end,
                                                     counts_.begin() + c_offset,
                                                     counts_.begin() + c_offset + c_size);
                        }

                        if(match) {
                            new_entry = false;
                            new_offset_dict[entry] = c_offset;
                            break;
                        }
                    }
                }

                // if this is a new entry, add it to the ids / counts, offsets / sizes and hashes
                if(new_entry) {

                    const std::size_t this_size = std::distance(ids_begin, ids_end);
                    const std::size_t this_offset = ids_.size();
                    offsets_.emplace_back(this_offset);
                    entry_sizes_.emplace_back(this_size);
                    new_offset_dict[entry] = this_offset;

                    ids_.insert(ids_.end(), ids_begin, ids_end);
                    counts_.insert(counts_.end(), counts_begin, counts_end);

                    const OffsetType this_offset_id = offsets_.size();
                    if(hash_it == hashed_.end()) {
                        hashed_.emplace(hash, HashElement({this_offset_id}));
                    } else {
                        hash_it->second.emplace_back(this_offset_id);
                    }
                }
            }

            // update the offsets
            // note that we switch from entry based offsets to element based offsets here
            for(std::size_t ii = 0; ii < offsets.size(); ++ii) {
                offsets(ii) = new_offset_dict[offsets(ii)];
            }
        }

    private:
        // we assume that all entries are unique when
        // passed to the constructor
        inline void init_hashed() {
            const std::size_t n_entries = offsets_.size();
            for(std::size_t entry = 0; entry < n_entries; ++entry) {
                const OffsetType off = offsets_[entry];
                const OffsetType size = entry_sizes_[entry];

                auto max_it = std::max_element(counts_.begin() + off,
                                               counts_.begin() + off + size);
                const IdType max_label = ids_[std::distance(counts_.begin(), max_it)];
                const CountType max_count = *max_it;

                HashKey hash(max_label, max_count);
                auto hash_it = hashed_.find(hash);
                if(hash_it == hashed_.end()) {
                    hashed_.emplace(hash, HashElement({entry}));
                } else {
                    hash_it->second.emplace_back(entry);
                }
            }
        }

    //
    private:
        std::vector<OffsetType> offsets_;
        std::vector<OffsetType> entry_sizes_;
        std::vector<IdType> ids_;
        std::vector<CountType> counts_;
        HashDictType hashed_;
    };

}
}
