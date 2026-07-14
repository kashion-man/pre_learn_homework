#include "../exercise.h"
#include <cstring>
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        std::memcpy(shape,shape_,4 * sizeof(unsigned int));
        size = shape[0] * shape[1] * shape[2] * shape[3];
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        for (int i = 0; i < 4; i++) {
            if (this->shape[i] != others.shape[i] && others.shape[i] != 1) {
                // 如果维度不同，且 others 的该维度不是 1，则无法广播
                // 注意：由于返回引用，通常在实际工程中选择抛出异常，这里仅作防护
                return *this; 
            }
        }

        // 2. 计算 this 和 others 的各维度的累乘步长（Strides），方便后续坐标转换
        unsigned int this_sz3 = this->shape[3];
        unsigned int this_sz23 = this->shape[2] * this_sz3;
        unsigned int this_sz123 = this->shape[1] * this_sz23;
        unsigned int total_size = this->shape[0] * this_sz123;

        unsigned int oth_sz3 = others.shape[3];
        unsigned int oth_sz23 = others.shape[2] * oth_sz3;
        unsigned int oth_sz123 = others.shape[1] * oth_sz23;

        // 3. 遍历自身的所有元素
        for (unsigned int idx = 0; idx < total_size; ++idx) {
            // 将一维索引还原为 this 的 4D 坐标 (i0, i1, i2, i3)
            unsigned int i0 = idx / this_sz123;
            unsigned int rem = idx % this_sz123;
            unsigned int i1 = rem / this_sz23;
            rem %= this_sz23;
            unsigned int i2 = rem / this_sz3;
            unsigned int i3 = rem % this_sz3;

            // 映射到 others 的坐标：如果 others 的维度是 1，则坐标强转为 0（即广播）
            unsigned int o0 = (others.shape[0] == 1) ? 0 : i0;
            unsigned int o1 = (others.shape[1] == 1) ? 0 : i1;
            unsigned int o2 = (others.shape[2] == 1) ? 0 : i2;
            unsigned int o3 = (others.shape[3] == 1) ? 0 : i3;

            // 计算 others 在一维内存中的对应索引
            unsigned int others_idx = o0 * oth_sz123 + o1 * oth_sz23 + o2 * oth_sz3 + o3;

            // 执行加法
            this->data[idx] += others.data[others_idx];
        }

        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
