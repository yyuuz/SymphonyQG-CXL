#include <iostream>
#include <string>

#include "common.hpp"
#include "index/qg/qg.hpp"
#include "index/qg/qg_builder.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"

std::string dataset = "msong";
size_t degree = 32;
auto data_file = "./data/" + dataset + "/" + dataset + "_base.fvecs";
auto index_file = "./data/" + dataset + "/" + "symqg" + std::to_string(degree) + ".index";

int main() {
    using data_type = symqg::RowMatrix<float>;

    data_type data;

    symqg::load_vecs<float, data_type>(data_file.c_str(), data);

    StopW stopw;

    symqg::QuantizedGraph qg(data.rows(), 32, data.cols());

    symqg::QGBuilder builder(qg, 400, data.data(), 9999);

    // 3 iters, refine at last iter
    builder.build(false);
    builder.build(false);
    builder.build(true);

    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    qg.save(index_file.c_str());

    return 0;
}