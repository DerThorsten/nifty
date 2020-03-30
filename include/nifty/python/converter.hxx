#pragma once

#include <cctype>
#include <type_traits>
#include <initializer_list>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/block_access.hxx"

namespace py = pybind11;

namespace nifty{

    inline std::string lowerFirst(const std::string & name){
        auto r = name;
        r[0] = std::tolower(name[0]);
        return r;
    }
}

namespace pybind11{
namespace detail{

    // to avoid the include of pybind/stl.h
    // we re-implement the array_caster
    template <typename ArrayType, typename Value,
              bool Resizable, std::size_t Size = 0> struct array_caster_ {
        using value_conv = make_caster<Value>;

    private:
        template <bool R = Resizable>
        bool require_size(enable_if_t<R, std::size_t> size) {
            if (value.size() != size)
                value.resize(size);
            return true;
        }
        template <bool R = Resizable>
        bool require_size(enable_if_t<!R, std::size_t> size) {
            return size == Size;
        }

    public:
        bool load(handle src, bool convert) {
            if (!isinstance<list>(src))
                return false;
            auto l = reinterpret_borrow<list>(src);
            if (!require_size(l.size()))
                return false;
            value_conv conv;
            std::size_t ctr = 0;
            for (auto it : l) {
                if (!conv.load(it, convert))
                    return false;
                value[ctr++] = cast_op<Value>(conv);
            }
            return true;
        }

        static handle cast(const ArrayType &src, return_value_policy policy, handle parent) {
            list l(src.size());
            std::size_t index = 0;
            for (auto const &value: src) {
                auto value_ = reinterpret_steal<object>(value_conv::cast(value, policy, parent));
                if (!value_)
                    return handle();
                PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); // steals a reference
            }
            return l.release();
        }

        PYBIND11_TYPE_CASTER(ArrayType, _("List[") + value_conv::name + _<Resizable>(_(""), _("[") + _<Size>() + _("]")) + _("]"));
    };


    template <typename Type, std::size_t Size>
    struct type_caster< nifty::array::StaticArray<Type, Size> >
    : array_caster_< nifty::array::StaticArray<Type, Size>, Type, false, Size> {
    };
}
}
