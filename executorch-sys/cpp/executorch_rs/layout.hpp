#include <cstddef>

namespace executorch_rs
{
    struct Layout
    {
        size_t size;
        size_t alignment;
    };
    template <typename T>
    static constexpr Layout layout_of()
    {
        return Layout{
            .size = sizeof(T),
            .alignment = alignof(T),
        };
    }
    constexpr bool operator==(const Layout &lhs, const Layout &rhs)
    {
        return lhs.size == rhs.size && lhs.alignment == rhs.alignment;
    }

    template <typename T, typename U>
    constexpr bool is_equal_layout()
    {
        return layout_of<T>() == layout_of<U>();
    }

    template <typename U, typename T>
    constexpr U *checked_reinterpret_cast(T *ptr)
    {
        static_assert(executorch_rs::is_equal_layout<T, U>());
        return reinterpret_cast<U *>(ptr);
    }
    template <typename U, typename T>
    constexpr U **checked_reinterpret_cast(T **ptr)
    {
        static_assert(executorch_rs::is_equal_layout<T, U>());
        return reinterpret_cast<U **>(ptr);
    }

    template <typename U, typename T>
    constexpr const U *checked_reinterpret_cast(const T *ptr)
    {
        static_assert(executorch_rs::is_equal_layout<T, U>());
        return reinterpret_cast<const U *>(ptr);
    }
    template <typename U, typename T>
    constexpr const U **checked_reinterpret_cast(const T **ptr)
    {
        static_assert(executorch_rs::is_equal_layout<T, U>());
        return reinterpret_cast<const U **>(ptr);
    }
}