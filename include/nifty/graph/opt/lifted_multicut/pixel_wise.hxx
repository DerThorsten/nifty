#pragma once


#include <iostream>

#include <boost/container/flat_map.hpp>

#include <xtensor/xoperation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xview.hpp>

#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{


    ///\cond
    namespace detail_plmc{
        template<std::size_t DIM>
        struct EvalHelper;

        template<>
        struct EvalHelper<2>{

            template<class OBJ, class D_LABELS>
            static auto evaluate(
                const OBJ & obj,
                const xt::xexpression<D_LABELS> & e_labels
            ){
                const auto & labels = e_labels.derived_cast();
                const auto & weights = obj.weights();
                const auto & offsets = obj.offsets();
                const auto & shape = obj.shape();
                const auto & n_offsets = obj.n_offsets();

                auto e = 0.0;

                for(int p0=0; p0<shape[0]; ++p0)
                for(int p1=0; p1<shape[1]; ++p1){
                    const auto label_p = labels(p0, p1);
                    for(int offset_index=0; offset_index<n_offsets; ++offset_index){
                        const int q0 = p0 + offsets(offset_index, 0);
                        const int q1 = p1 + offsets(offset_index, 1);
                        if(q0 >= 0 && q0 < shape[0]  && q1 >= 0 && q1 < shape[1]){
                            const auto label_q = labels(q0, q1);
                            if(label_p != label_q){
                                e += weights(p0, p1, offset_index);
                            }
                        }
                    }
                }
                return e;
            }
        };
    }
    ///\endcond







    template<class D_WEIGHTS, class D_OFFSETS>
    auto pixel_wise_lmc_edge_gt_2d(
        const xt::xexpression<D_WEIGHTS> & e_gt,
        const xt::xexpression<D_OFFSETS> & e_offsets
    ){
        typedef xt::xtensor<bool, 3, xt::layout_type::row_major> ReturnType;
       

        const auto & gt = e_gt.derived_cast();
        const auto & offsets = e_offsets.derived_cast();
        const auto & shape = gt.shape();
        const auto n_offsets = offsets.shape()[0];

        typename ReturnType::shape_type result_shape{
            size_t(shape[0]), size_t(shape[1]), size_t(n_offsets)};

        auto result = ReturnType(result_shape);

        // fill the lifted obj
        for(int p0=0; p0<shape[0]; ++p0)
        for(int p1=0; p1<shape[1]; ++p1){

            const auto p_label = gt(p0, p1);

            for(int offset_index=0; offset_index<n_offsets; ++offset_index){
                const int q0 = p0 + offsets(offset_index, 0);
                const int q1 = p1 + offsets(offset_index, 1);
                if(q0 >= 0 && q0 < shape[0]  && q1 >= 0 && q1 < shape[1]){
                    const auto q_label = gt(q0, q1);
                    result(p0, p1, offset_index) = p_label != q_label;
                }
            }
        }
        return result;
    }







    template<std::size_t DIM>
    class PixelWiseLmcObjective{
    public:

        PixelWiseLmcObjective(
        ) {

        }
        template<class D_WEIGHTS, class D_OFFSETS>
        PixelWiseLmcObjective(
            const xt::xexpression<D_WEIGHTS> & e_weights,
            const xt::xexpression<D_OFFSETS> & e_offsets
        )   
        :   weights_(e_weights),
            offsets_(e_offsets),
            shape_(),
            n_offsets_(),
            n_variables_()
        {
            // shape and n_offset
            const auto & wshape = weights_.shape();
            std::copy(wshape.begin(), wshape.end(), shape_.begin());
            n_offsets_ = wshape[DIM];

            // n_var 
            n_variables_ = shape_[0];
            for(auto d=1; d<DIM; ++d){
                n_variables_ *= shape_[d];
            }

        }

        template<class D_LABELS>
        auto evaluate(
            const xt::xexpression<D_LABELS> & e_labels
        )const{
            return detail_plmc::EvalHelper<DIM>::evaluate(*this, e_labels);
        }
        const auto & weights()const{
            return weights_;
        }
        const auto & offsets()const{
            return offsets_;
        }
        const auto & shape()const{
            return shape_;
        }
        auto n_offsets()const{
            return n_offsets_;
        }
        auto n_variables() const {
            return n_variables_;
        }
    private:

        xt::xtensor<int,       2, xt::layout_type::row_major> offsets_;
        xt::xtensor<float, DIM+1, xt::layout_type::row_major> weights_;

        std::array<int, DIM> shape_;
        std::size_t n_offsets_;
        uint64_t n_variables_;
    };






    template<std::size_t DIM>
    class PixelWiseLmcConnetedComponentsFusion;

    template<>
    class PixelWiseLmcConnetedComponentsFusion<2>
    {
    public:
        static const std::size_t DIM = 2;



        typedef nifty::graph::UndirectedGraph<>                 CCGraphType;
        typedef LiftedMulticutObjective<CCGraphType, double >   CCObjectiveType;
        typedef LiftedMulticutBase<CCObjectiveType>             CCBaseType;
        typedef typename CCBaseType::VisitorBaseType            CVisitorBaseType;
        typedef typename CCBaseType::NodeLabelsType             CCNodeLabels;
        // factory for the lifted primal rounder
        typedef nifty::graph::opt::common::SolverFactoryBase<CCBaseType> CCLmcFactoryBase;








        PixelWiseLmcConnetedComponentsFusion(
            const PixelWiseLmcObjective<DIM> & objective,
            std::shared_ptr<CCLmcFactoryBase> solver_fatory
        )
        :   objective_(objective),
            ufd_(objective.n_variables()),
            solver_fatory_(solver_fatory)
        {

        }


        struct  Settings
        {
            
        };


        

        template<class D_LABELS_A, class D_LABELS_B>
        auto fuse(
            const xt::xexpression<D_LABELS_A>  & e_labels_a,
            const xt::xexpression<D_LABELS_B>  & e_labels_b
        ){

            ufd_.reset();

            const auto & shape = objective_.shape();
            const auto & labels_a = e_labels_a.derived_cast();
            const auto & labels_b = e_labels_b.derived_cast();

            typename xt::xtensor<int, DIM>::shape_type reshape{size_t(shape[0]), size_t(shape[1])};
            auto res = xt::xtensor<int, DIM, xt::layout_type::row_major>(reshape);


            this->merge_ufd(e_labels_a, e_labels_b);

            // 
            auto e_a = objective_.evaluate(labels_a);
            auto e_b = objective_.evaluate(labels_b);
            this->do_it(res, [&](
                const auto & cc_node_labels,
                const auto & cc_energy
            ){
                if(cc_energy < std::min(e_a, e_b)){

                    auto res_iter = res.begin();
                    for(auto var=0; var<objective_.n_variables(); ++var){
                        const auto dense_var = *res_iter;
                        *res_iter = cc_node_labels[dense_var];
                        //*res_iter = cc_node_labels[to_dense[ufd_.find(var)]];
                        ++res_iter;
                    }   

                }
                else if(e_a < cc_energy){
                    std::copy(labels_a.begin(), labels_a.end(), res.begin());
                }
                else{
                    std::copy(labels_b.begin(), labels_b.end(), res.begin());
                }
            });

            return res;

        }


        template<class D_LABELS>
        auto fuse(
            const xt::xexpression<D_LABELS>  & e_labels
        ){



            ufd_.reset();

            const auto & shape = objective_.shape();





            typename xt::xtensor<int, DIM>::shape_type reshape{size_t(shape[0]), size_t(shape[1])};
            auto res = xt::xtensor<int, DIM, xt::layout_type::row_major>(reshape);


            this->merge_ufd2(e_labels);


            // 
           
            this->do_it(res, [&](
                const auto & cc_node_labels,
                const auto & cc_energy
            ){

                const auto & labels = e_labels.derived_cast();
                const auto n_proposals = labels.shape()[DIM];
                auto best_e = std::numeric_limits<float>::infinity();
                auto best_i = 0;

                for(auto i=0; i<n_proposals; ++i){
                    const auto l = xt::view(e_labels.derived_cast(),xt::all(), xt::all(),i);
                    const auto e = objective_.evaluate(l);
                    if(e < best_e){
                        best_e = e;
                        best_i = i;
                    }
                }
                

                if(cc_energy < best_e){//cc_energy < std::min(e_a, e_b)){
                    
                    auto res_iter = res.begin();
                    for(auto var=0; var<objective_.n_variables(); ++var){
                        const auto dense_var = *res_iter;
                        *res_iter = cc_node_labels[dense_var];
                        //*res_iter = cc_node_labels[to_dense[ufd_.find(var)]];
                        ++res_iter;
                    }   

                }
                else{
                    const auto l = xt::view(e_labels.derived_cast(),xt::all(), xt::all(),best_i);
                    std::copy(l.begin(), l.end(), res.begin());
                }
            });

            return res;

        }











    private:


        template<class D_LABELS_A, class D_LABELS_B>
        auto merge_ufd(
            const xt::xexpression<D_LABELS_A>  & e_labels_a,
            const xt::xexpression<D_LABELS_B>  & e_labels_b
        ){
            const auto & shape = objective_.shape();
            const auto & labels_a = e_labels_a.derived_cast();
            const auto & labels_b = e_labels_b.derived_cast();

            uint64_t node_p = 0;
            for(int p0=0; p0<shape[0]; ++p0)
            for(int p1=0; p1<shape[1]; ++p1){

                const auto p_label_a = labels_a(p0, p1);
                const auto p_label_b = labels_b(p0, p1);

                if(p0 + 1 < shape[0]){
                    const auto q_label_a = labels_a(p0+1, p1);
                    const auto q_label_b = labels_b(p0+1, p1);
                    if(p_label_a == q_label_a && p_label_b == q_label_b){
                        const auto node_q = node_p + shape[1];
                        ufd_.merge(node_p, node_q);
                    }
                }
                if(p1 + 1 < shape[1]){
                    const auto q_label_a = labels_a(p0, p1+1);
                    const auto q_label_b = labels_b(p0, p1+1);
                    if(p_label_a == q_label_a && p_label_b == q_label_b){
                        const auto node_q = node_p + 1;
                        ufd_.merge(node_p, node_q);
                    }
                }
                ++node_p;
            }
        }




        template<class D_LABELS>
        auto merge_ufd2(
            const xt::xexpression<D_LABELS>  & e_labels
        ){
            const auto & shape = objective_.shape();
            const auto & labels = e_labels.derived_cast();
            const auto n_offsets = labels.shape()[DIM];

            //const auto pview = xt::view(labels,0,0,xt::all());
            //const auto bla = pview(0);

            uint64_t node_p = 0;
            for(int p0=0; p0<shape[0]; ++p0)
            for(int p1=0; p1<shape[1]; ++p1){
                if(p0 + 1 < shape[0]){
                    bool do_merge = true;
                    for(auto o=0; o<n_offsets; ++o){
                        const auto p_label = labels(p0,  p1,o);
                        const auto q_label = labels(p0+1,p1,o);
                        if(p_label != q_label ){ 
                            do_merge = false;
                            break;
                        }
                    }
                    if(do_merge){
                        const auto node_q = node_p + shape[1];
                        ufd_.merge(node_p, node_q);
                    }
                }
                if(p1 + 1 < shape[1]){
                    bool do_merge = true;
                    for(auto o=0; o<n_offsets; ++o){
                        const auto p_label = labels(p0, p1,  o);
                        const auto q_label = labels(p0, p1+1,o);
                        if(p_label != q_label ){ 
                            do_merge = false;
                            break;
                        }
                    }
                    if(do_merge){
                        const auto node_q = node_p + 1;
                        ufd_.merge(node_p, node_q);
                    }
                }
                ++node_p;
            }
        }



        template<class F>
        auto do_it(
            xt::xtensor<int, DIM, xt::layout_type::row_major> & res,
            F && f
        ){


            const auto & shape = objective_.shape();
            const auto & offsets = objective_.offsets();
            const auto & weights = objective_.weights();
            const auto & n_offsets = objective_.n_offsets();
            // const auto & labels_a = e_labels_a.derived_cast();
            // const auto & labels_b = e_labels_b.derived_cast();


            // make map dense   
            const auto cc_n_variables = ufd_.numberOfSets();
            boost::container::flat_map<uint64_t, uint64_t> to_dense;
            ufd_.representativeLabeling(to_dense);
            {
                auto res_iter = res.begin();
                for(auto var=0; var<objective_.n_variables(); ++var){
                    *res_iter = to_dense[ufd_.find(var)];
                    ++res_iter;
                }   
            }



            
            // build the normal graph
            CCGraphType cc_graph(cc_n_variables);

            uint64_t node_p = 0 ;
            for(int p0=0; p0<shape[0]; ++p0)
            for(int p1=0; p1<shape[1]; ++p1){

                const auto p_label = ufd_.find(node_p);

                if(p0 + 1 < shape[0]){
                    const auto node_q = node_p + shape[1];
                    const auto q_label = ufd_.find(node_q);
                    if(p_label != q_label){
                        cc_graph.insertEdge(to_dense[p_label],to_dense[q_label]);
                    }
                }
                if(p1 + 1 < shape[1]){
                    const auto node_q = node_p + 1;
                    const auto q_label = ufd_.find(node_q);
                    if(p_label != q_label){
                        cc_graph.insertEdge(to_dense[p_label],to_dense[q_label]);
                    }
                }
                ++node_p;
            }

            CCObjectiveType cc_obj(cc_graph);

            node_p = 0 ;

            // fill the lifted obj
            for(int p0=0; p0<shape[0]; ++p0)
            for(int p1=0; p1<shape[1]; ++p1){

                const auto p_label = ufd_.find(node_p);

                for(int offset_index=0; offset_index<n_offsets; ++offset_index){
                    const int q0 = p0 + offsets(offset_index, 0);
                    const int q1 = p1 + offsets(offset_index, 1);
                    if(q0 >= 0 && q0 < shape[0]  && q1 >= 0 && q1 < shape[1]){

                        const auto node_q = q0*shape[1] + q1;
                        const auto q_label = ufd_.find(node_q);
                        if(p_label != q_label){

                            cc_obj.setCost(to_dense[p_label], to_dense[q_label], 
                                weights(p0,p1,offset_index));
                        }
                    }
                }
                ++node_p;
            }


            auto solver = solver_fatory_->create(cc_obj);
            CCNodeLabels cc_node_labels(cc_graph);


            nifty::graph::opt::common::VerboseVisitor<CCBaseType> visitor;
            solver->optimize(cc_node_labels, nullptr);
            auto e_res = cc_obj.evalNodeLabels(cc_node_labels);
            delete solver;  
            f(cc_node_labels, e_res);
        }


        const PixelWiseLmcObjective<DIM> & objective_;
        nifty::ufd::Ufd<uint64_t> ufd_;
        std::shared_ptr<CCLmcFactoryBase>  solver_fatory_;
    };



} // namespace lifted_multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

