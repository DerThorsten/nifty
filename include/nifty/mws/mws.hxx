#pragma once
#include "nifty/ufd/ufd.hxx"
#include "nifty/xtensor/xtensor.hxx"

namespace nifty {
namespace mws {


    // the datastructure to hold the mutex edges for a single cluster and all clusters
    typedef std::unordered_set<uint32_t> MutexSet;
    typedef std::vector<MutexSet> MutexStorage;


    template<class UFD>
    inline bool check_mutex(const uint32_t u, const uint32_t rv,
                            UFD & ufd, const MutexStorage & mutexes) {
        // the mutex storages are symmetric, so we only need to check one of them
        const auto & mutex_u = mutexes[u];
        bool have_mutex = false;
        // we check for all representatives of mutex edges if
        // they are the same as the reperesentative of v
        for(const auto mu : mutex_u) {
            if(ufd.find(mu) == rv) {
                have_mutex = true;
                break;
            }
        }
        return have_mutex;
    }


    template<class UFD>
    inline void insert_mutex(const uint32_t u, const uint32_t v, const uint32_t rv,
                             UFD & ufd, MutexStorage & mutexes) {

        auto & mutex_u = mutexes[u];
        // if we don't have a mutex yet, insert it
        if(mutex_u.size() == 0) {
            mutex_u.insert(v);
        }

        // otherwise check if v is already in the mutexes
        // and filter the mutexes in the process
        else {

            bool have_mutex = false;
            std::unordered_set<uint32_t> mutex_representatives;

            // iterate over all current mutexes
            auto mutex_it = mutex_u.begin();
            while(mutex_it != mutex_u.end()) {
                const uint32_t rm = ufd.find(*mutex_it);

                // check if this mutex is already present in the list
                // if it is not, insert it, otherwise delete this mutex
                if(mutex_representatives.find(rm) == mutex_representatives.end()) {
                    mutex_representatives.insert(rm);
                    ++mutex_it;  // we don't erase, so we need to increase by hand
                } else {
                    mutex_it = mutex_u.erase(mutex_it);
                }

                // if we have not already found v as mutex, check for it
                if(!have_mutex) {
                    have_mutex = rv == rm;
                }
            }

            // insert the v mutex if it is not present
            if(!have_mutex) {
                // std::cout << "Inserting mutex " << u << " " << v << std::endl;
                mutex_u.insert(v);
            }
        }
    }


    template<class UFD>
    inline void merge_mutexes(const uint32_t u, const uint32_t v, UFD & ufd, MutexStorage & mutexes) {
        auto & mutex_u = mutexes[u];
        auto & mutex_v = mutexes[v];

        // extract all representatives (which should be unique here)
        std::unordered_map<uint32_t, uint32_t> mutex_reps_u;
        for(const auto mu : mutex_u) {
            mutex_reps_u[ufd.find(mu)] = mu;
        }

        std::unordered_map<uint32_t, uint32_t> mutex_reps_v;
        for(const auto mv : mutex_v) {
            mutex_reps_v[ufd.find(mv)] = mv;
        }

        // merge u into v
        for(const auto mu : mutex_reps_u) {
            if(mutex_reps_v.find(mu.first) == mutex_reps_v.end()) {
                mutex_v.insert(mu.second);
            }
        }

        // merge v into u
        for(const auto mv : mutex_reps_v) {
            if(mutex_reps_u.find(mv.first) == mutex_reps_u.end()) {
                mutex_u.insert(mv.second);
            }
        }

    }


    // TODO this should be a class
    // FIXME naive first implementation
    template<class EDGE_ARRAY, class WEIGHT_ARRAY, class NODE_ARRAY>
    void computeMwsClustering(const uint32_t number_of_labels,
                              const xt::xexpression<EDGE_ARRAY> & uvs_exp,
                              const xt::xexpression<EDGE_ARRAY> & mutex_uvs_exp,
                              const xt::xexpression<WEIGHT_ARRAY> & weights_exp,
                              const xt::xexpression<WEIGHT_ARRAY> & mutex_weights_exp,
                              xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & uvs = uvs_exp.derived_cast();
        const auto & mutex_uvs = mutex_uvs_exp.derived_cast();
        const auto & weights = weights_exp.derived_cast();
        const auto & mutex_weights = mutex_weights_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        // make ufd
        ufd::Ufd<uint32_t> ufd(number_of_labels);

        // determine number of edge types
        const size_t num_edges = uvs.shape()[0];
        const size_t num_mutex = mutex_uvs.shape()[0];

        // argsort ALL edges
        // we sort in ascending order (TODO is this correct ?)
        std::vector<size_t> indices(num_edges + num_mutex);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
            const double val_a = (a < num_edges) ? weights(a) : mutex_weights(a - num_edges);
            const double val_b = (b < num_edges) ? weights(b) : mutex_weights(b - num_edges);
            return val_a < val_b;
        });

        MutexStorage mutexes(number_of_labels);

        // iterate over all edges
        for(const size_t edge_id : indices) {

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= num_edges;

            if(is_mutex) {
                // find the mutex id and the connected nodes
                const size_t mutex_id = edge_id - num_edges;
                const uint32_t u = mutex_uvs(mutex_id, 0);
                const uint32_t v = mutex_uvs(mutex_id, 1);

                // find the current representatives
                const uint32_t ru = ufd.find(u);
                const uint32_t rv = ufd.find(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, insert the mutex
                insert_mutex(u, v, rv, ufd, mutexes);
                insert_mutex(v, u, ru, ufd, mutexes);
                // insert_mutex_edge(u, mutex_id, mutexes);
                // insert_mutex_edge(v, mutex_id, mutexes);

            } else {

                // find the connected nodes
                const uint32_t u = uvs(edge_id, 0);
                const uint32_t v = uvs(edge_id, 1);

                // find the current representatives
                const uint32_t ru = ufd.find(u);
                const uint32_t rv = ufd.find(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(u, rv, ufd, mutexes) || check_mutex(v, ru, ufd, mutexes);
                //const bool have_mutex = check_mutex_edge(u, v, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    ufd.merge(u, v);
                    merge_mutexes(u, v, ufd, mutexes);
                    //merge_mutex_edges(u, v, mutexes);
                }

            }
        }

        // get node labeling into output
        ufd.elementLabeling(node_labeling.begin());
    }

}
}
