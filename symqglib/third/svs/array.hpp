// The implementation is largely based on the implementation of SVS.
// https://github.com/intel/ScalableVectorSearch

/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <utility>

#include "../../utils/memory.hpp"



namespace symqg::data::array_impl {
/**
 * @brief get size of array
 */
template <typename Dims>
[[nodiscard]] constexpr auto size(const Dims& dims) -> size_t {
    static_assert(std::is_same_v<typename Dims::value_type, size_t>);

    size_t res = 1;
    std::for_each(dims.begin(), dims.end(), [&](auto cur_d) { res *= cur_d; });
    return res;
}
}  // namespace symqg::data::array_impl

namespace symqg::data {
template <
    typename T,
    typename Dims = std::vector<size_t>,
    typename Alloc = memory::Allocator<T>>
class Array {
   private:
    static_assert(std::is_trivial_v<T>);  // only handle trivial types

    /// @brief num of data objects
    [[nodiscard]] constexpr auto size() const -> size_t { return array_impl::size(dims_); }

    /// @brief num of bytes for all data objects
    [[nodiscard]] constexpr auto bytes() const -> size_t { return sizeof(T) * size(); }

    void destroy() {
        size_t num_elements = size();
        if(use_cxl_) {
            std::cout<<"Freeing with CXL\n";
            memkind_free(MEMKIND_DAX_KMEM_ALL, pointer_);
        }
        else {
            atraits::deallocate(allocator_, pointer_, num_elements);
        }
        pointer_ = nullptr;
    }

   public:
    using allocator_type = Alloc;
    using atraits = std::allocator_traits<allocator_type>;
    using pointer = typename atraits::pointer;
    using const_pointer = typename atraits::const_pointer;

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    Array() = default;

    explicit Array(Dims dims, const Alloc& allocator)
        : dims_(std::move(dims)), allocator_(allocator) {
        //allocator_ = memory::AlignedAllocator<float, 1 << 22, true>(true);
        size_t num_elements = size();
        pointer_ = atraits::allocate(allocator_, num_elements);
        use_cxl_ = allocator_.USE_CXL;
    }

    explicit Array(Dims dims) : Array(std::move(dims), Alloc()) {}

    ~Array() noexcept {
        if (pointer_ != nullptr) {
            destroy();
        }
    }

    /// @brief move constructor
    Array(Array&& other) noexcept
        : pointer_{std::exchange(other.pointer_, nullptr)}
        , dims_{std::move(other.dims_)}
        , use_cxl_{other.use_cxl_}
        , allocator_{std::move(other.allocator_)} {}

    Array& operator=(Array&& other) noexcept {
        if (pointer_ != nullptr) {
            destroy();
        }

        if constexpr (atraits::propagate_on_container_move_assignment::value) {
            allocator_ = std::move(other.allocator_);
        }
        dims_ = std::exchange(other.dims_, Dims());
        pointer_ = std::exchange(other.pointer_, nullptr);
        use_cxl_ = other.use_cxl_;
        return *this;
    }

    [[nodiscard]] pointer data() { return pointer_; }
    [[nodiscard]] const_pointer data() const { return pointer_; }

    [[nodiscard]] reference at(size_t idx) { return pointer_[idx]; }
    [[nodiscard]] const_reference at(size_t idx) const { return pointer_[idx]; }

    void save(std::ofstream& output) const {
        if (output.good()) {
            output.write(reinterpret_cast<char*>(pointer_), bytes());
        }
    }
    void load(std::ifstream& input) {
        input.read(reinterpret_cast<char*>(pointer_), bytes());
    }

    reference operator[](size_t idx) { return pointer_[idx]; }

   private:
    pointer pointer_ = nullptr;
    [[no_unique_address]] Dims dims_;
    [[no_unique_address]] Alloc allocator_;
    bool use_cxl_=false ;
};
}  // namespace symqg::data