// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_istft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a out center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length window=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = 0; // all ones
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft, 20)

class torch_istft_1 : public torch_istft
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a b center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length window=None
torch.view_as_real      op_2        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        torch_istft::write(op, captured_params);

        op->params["1"] = 0; // returns
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_1, 20)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int detect_window_type(const std::vector<float>& window_data)
{
    const int winlen = (int)window_data.size();

    bool is_one = true;
    bool is_hann = true;
    bool is_hamming = true;
    for (int i = 0; i < winlen; i++)
    {
        if (!NearlyEqual(window_data[i], 1.f, 0.001))
            is_one = false;

        if (!NearlyEqual(window_data[i], 0.5f * (1 - cos(2 * 3.14159265358979323846 * i / winlen)), 0.001))
            is_hann = false;

        if (!NearlyEqual(window_data[i], 0.54f - 0.46f * cos(2 * 3.14159265358979323846 * i / winlen), 0.001))
            is_hamming = false;
    }

    if (is_one)
        return 0;
    if (is_hann)
        return 1;
    if (is_hamming)
        return 2;

    return -1;
}

class torch_istft_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
pnnx.Attribute          op_1        0 1 window @data
torch.istft             op_2        2 1 a window out center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    bool match(const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_1.data").get_float32_data();
        const int window_type = detect_window_type(window_data);
        return window_type != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_1.data").get_float32_data();
        const int window_type = detect_window_type(window_data);

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_2, 20)

class torch_istft_3 : public torch_istft_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
pnnx.Attribute          op_1        0 1 window @data
torch.istft             op_2        2 1 a window b center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
torch.view_as_real      op_3        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        torch_istft_2::write(op, captured_params, captured_attrs);

        op->params["1"] = 0; // returns
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_3, 20)

} // namespace ncnn

} // namespace pnnx
