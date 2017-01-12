#pragma once
#ifndef NIFTY_PYTHON_GRAPH_ADJACENCY_CONCERTER_HXX
#define NIFTY_PYTHON_GRAPH_ADJACENCY_CONCERTER_HXX

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include <nifty/graph/detail/adjacency.hxx>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <typename T1, typename T2> class type_caster<nifty::graph::detail_graph::UndirectedAdjacency<T1, T2>> {
    typedef nifty::graph::detail_graph::UndirectedAdjacency<T1, T2> type;

public:
    bool load(handle src, bool convert) {
        if (!src)
            return false;
        else if (!PyTuple_Check(src.ptr()) || PyTuple_Size(src.ptr()) != 2)
            return false;
        return  first.load(PyTuple_GET_ITEM(src.ptr(), 0), convert) &&
                second.load(PyTuple_GET_ITEM(src.ptr(), 1), convert);
    }

    static handle cast(const type &src, return_value_policy policy, handle parent) {
        object o1 = object(type_caster<typename intrinsic_type<T1>::type>::cast(src.node(), policy, parent), false);
        object o2 = object(type_caster<typename intrinsic_type<T2>::type>::cast(src.edge(), policy, parent), false);
        if (!o1 || !o2)
            return handle();
        tuple result(2);
        PyTuple_SET_ITEM(result.ptr(), 0, o1.release().ptr());
        PyTuple_SET_ITEM(result.ptr(), 1, o2.release().ptr());
        return result.release();
    }

    static PYBIND11_DESCR name() {
        return type_descr(
            _("(") + type_caster<typename intrinsic_type<T1>::type>::name() +
            _(", ") + type_caster<typename intrinsic_type<T2>::type>::name() + _(")"));
    }

    template <typename T> using cast_op_type = type;

    operator type() {
        return type(first .operator typename type_caster<typename intrinsic_type<T1>::type>::template cast_op_type<T1>(),
                    second.operator typename type_caster<typename intrinsic_type<T2>::type>::template cast_op_type<T2>());
    }
protected:
    type_caster<typename intrinsic_type<T1>::type> first;
    type_caster<typename intrinsic_type<T2>::type> second;
};


} //namespace detail
} //namepace pybind11

#endif //NIFTY_PYTHON_GRAPH_ADJACENCY_CONCERTER_HXX
