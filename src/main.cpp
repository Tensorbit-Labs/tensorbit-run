#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "tensorbit/run/backend.hpp"
#include "tensorbit/run/common.hpp"
#include "tensorbit/run/common.h"
#include "tensorbit/run/model.h"
#include "tensorbit/run/model.hpp"
#include "tensorbit/run/ops.h"
#include "tensorbit/run/ops.hpp"
#include "tensorbit/run/tensor.hpp"

using namespace tensorbit::run;

/* ================================================================
 * Simple tokenizer: maps chars to IDs for demo purposes.
 * In production, use a proper BPE/SentencePiece tokenizer.
 * ================================================================ */

struct SimpleTokenizer {
    std::vector<std::string> vocab;
    int                      vocab_size = 0;

    void init_default() {
        vocab = {"<pad>", "<bos>", "<eos>", "<unk>"};
        for (int i = 0; i < 26; i++) {
            std::string s(1, (char)('a' + i));
            vocab.push_back(s);
        }
        for (int i = 0; i < 10; i++) {
            std::string s(1, (char)('0' + i));
            vocab.push_back(s);
        }
        vocab.push_back(" ");
        vocab.push_back(".");
        vocab.push_back(",");
        vocab.push_back("!");
        vocab.push_back("?");
        vocab.push_back("'");
        vocab.push_back("\"");
        vocab.push_back("-");
        vocab.push_back(":");
        vocab.push_back(";");
        vocab_size = (int)vocab.size();
    }

    int encode_char(char c) {
        std::string s(1, c);
        for (int i = 0; i < vocab_size; i++) {
            if (vocab[i] == s) return i;
        }
        return 3; /* <unk> */
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        for (char c : text) {
            ids.push_back(encode_char(c));
        }
        return ids;
    }

    std::string decode(int id) {
        if (id >= 0 && id < vocab_size) return vocab[id];
        return "<unk>";
    }
};

/* ================================================================
 * Sampling: argmax from logits
 * ================================================================ */

static int sample_argmax(const float* logits, int n) {
    int   best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static int sample_temperature(const float* logits, int n, float temperature) {
    if (temperature <= 0.0f) return sample_argmax(logits, n);

    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    std::vector<double> probs(n);
    double              sum = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp((double)((logits[i] - max_val) / temperature));
        sum += probs[i];
    }

    for (int i = 0; i < n; i++) probs[i] /= sum;

    static std::mt19937                          rng(42);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

/* ================================================================
 * Mock model creation — creates a tiny .tbm for testing
 * ================================================================ */

static int create_mock_tbm(const char* path, int hidden_size, int intermediate_size, int num_layers,
                           int vocab_size, int max_seq_len) {
    int n_heads = 8;
    int n_kv_heads = 8;
    int head_dim = hidden_size / n_heads;

    TbModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strcpy(cfg.architecture, "llama");
    cfg.hidden_size = hidden_size;
    cfg.num_heads = n_heads;
    cfg.num_kv_heads = n_kv_heads;
    cfg.intermediate_size = intermediate_size;
    cfg.vocab_size = vocab_size;
    cfg.num_layers = num_layers;
    cfg.max_seq_len = max_seq_len;
    cfg.norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;

    /* Collect all layers */
    std::vector<TbLayerInput> layer_inputs;

    /* Embedding */
    {
        size_t nw = (size_t)vocab_size * hidden_size;
        size_t nb = nw * sizeof(float);
        float* w = (float*)tb_malloc(nb);
        for (size_t i = 0; i < nw; i++)
            w[i] = ((float)(rand()) / RAND_MAX - 0.5f) * 0.02f;

        TbLayerInput li;
        memset(&li, 0, sizeof(li));
        strcpy(li.name, "model.embed_tokens.weight");
        li.shape[0] = vocab_size;
        li.shape[1] = hidden_size;
        li.nm_n = 0;
        li.nm_m = 0;
        li.dtype = TB_DTYPE_F32;
        li.weight_data = w;
        li.weight_byte_size = nb;
        li.mask_data = NULL;
        li.mask_byte_size = 0;
        layer_inputs.push_back(li);
    }

    /* Per-layer tensors */
    for (int l = 0; l < num_layers; l++) {
        /* Q, K, V, O projections */
        struct {
            const char* name;
            size_t      out_dim;
            size_t      in_dim;
            int         nm;
            int         mm;
        } projs[] = {
            {"model.layers.%d.self_attn.q_proj.weight", (size_t)(n_heads * head_dim),
             (size_t)hidden_size, 2, 4},
            {"model.layers.%d.self_attn.k_proj.weight", (size_t)(n_kv_heads * head_dim),
             (size_t)hidden_size, 2, 4},
            {"model.layers.%d.self_attn.v_proj.weight", (size_t)(n_kv_heads * head_dim),
             (size_t)hidden_size, 2, 4},
            {"model.layers.%d.self_attn.o_proj.weight", (size_t)hidden_size,
             (size_t)(n_heads * head_dim), 2, 4},
            {"model.layers.%d.mlp.gate_proj.weight", (size_t)intermediate_size, (size_t)hidden_size,
             2, 4},
            {"model.layers.%d.mlp.up_proj.weight", (size_t)intermediate_size, (size_t)hidden_size, 2,
             4},
            {"model.layers.%d.mlp.down_proj.weight", (size_t)hidden_size,
             (size_t)intermediate_size, 2, 4},
        };

        for (auto& p : projs) {
            char   name_buf[256];
            snprintf(name_buf, sizeof(name_buf), p.name, l);

            size_t nw = p.out_dim * p.in_dim;
            size_t nb = nw * sizeof(float);
            float* w = (float*)tb_malloc(nb);
            for (size_t i = 0; i < nw; i++)
                w[i] = ((float)(rand()) / RAND_MAX - 0.5f) * 0.02f;

            size_t groups = (p.in_dim / p.mm) * p.out_dim;
            size_t mask_bytes = groups;
            uint8_t* mask = (uint8_t*)tb_malloc(mask_bytes);
            if (p.mm > 0 && p.nm > 0) {
                for (size_t g = 0; g < groups; g++) {
                    uint8_t m = 0;
                    for (int b = 0; b < p.nm; b++) m |= (uint8_t)(1u << b);
                    mask[g] = m;
                }
            } else {
                memset(mask, 0, mask_bytes);
            }

            TbLayerInput li;
            memset(&li, 0, sizeof(li));
            strcpy(li.name, name_buf);
            li.shape[0] = p.out_dim;
            li.shape[1] = p.in_dim;
            li.nm_n = p.nm;
            li.nm_m = p.mm;
            li.dtype = TB_DTYPE_F32;
            li.weight_data = w;
            li.weight_byte_size = nb;
            li.mask_data = mask;
            li.mask_byte_size = mask_bytes;
            layer_inputs.push_back(li);
        }

        /* RMSNorm weights */
        {
            const char* ln_names[] = {"model.layers.%d.input_layernorm.weight",
                                       "model.layers.%d.post_attention_layernorm.weight"};

            for (auto& ln_name : ln_names) {
                char   name_buf[256];
                snprintf(name_buf, sizeof(name_buf), ln_name, l);

                size_t nb = hidden_size * sizeof(float);
                float* w = (float*)tb_malloc(nb);
                for (int i = 0; i < hidden_size; i++) w[i] = 1.0f;

                TbLayerInput li;
                memset(&li, 0, sizeof(li));
                strcpy(li.name, name_buf);
                li.shape[0] = hidden_size;
                li.shape[1] = 0;
                li.nm_n = 0;
                li.nm_m = 0;
                li.dtype = TB_DTYPE_F32;
                li.weight_data = w;
                li.weight_byte_size = nb;
                li.mask_data = NULL;
                li.mask_byte_size = 0;
                layer_inputs.push_back(li);
            }
        }
    }

    /* Final norm */
    {
        size_t nb = hidden_size * sizeof(float);
        float* w = (float*)tb_malloc(nb);
        for (int i = 0; i < hidden_size; i++) w[i] = 1.0f;

        TbLayerInput li;
        memset(&li, 0, sizeof(li));
        strcpy(li.name, "model.norm.weight");
        li.shape[0] = hidden_size;
        li.shape[1] = 0;
        li.nm_n = 0;
        li.nm_m = 0;
        li.dtype = TB_DTYPE_F32;
        li.weight_data = w;
        li.weight_byte_size = nb;
        li.mask_data = NULL;
        li.mask_byte_size = 0;
        layer_inputs.push_back(li);
    }

    /* LM head */
    {
        size_t nw = (size_t)hidden_size * vocab_size;
        size_t nb = nw * sizeof(float);
        float* w = (float*)tb_malloc(nb);
        for (size_t i = 0; i < nw; i++)
            w[i] = ((float)(rand()) / RAND_MAX - 0.5f) * 0.02f;

        TbLayerInput li;
        memset(&li, 0, sizeof(li));
        strcpy(li.name, "lm_head.weight");
        li.shape[0] = vocab_size;
        li.shape[1] = hidden_size;
        li.nm_n = 0;
        li.nm_m = 0;
        li.dtype = TB_DTYPE_F32;
        li.weight_data = w;
        li.weight_byte_size = nb;
        li.mask_data = NULL;
        li.mask_byte_size = 0;
        layer_inputs.push_back(li);
    }

    int ret = tb_tbm_create(path, &cfg, layer_inputs.data(), (int)layer_inputs.size());

    /* Free temporary allocations */
    for (auto& li : layer_inputs) {
        if (li.weight_data) tb_free((void*)li.weight_data);
        if (li.mask_data) tb_free((void*)li.mask_data);
    }

    return ret;
}

/* ================================================================
 * Print usage
 * ================================================================ */

static void print_usage(const char* prog) {
    fprintf(stderr,
            "tensorbit-run — bare-metal C++20 inference engine\n"
            "\n"
            "Usage: %s [OPTIONS]\n"
            "\n"
            "Options:\n"
            "  --model <path>       .tbm model file to load\n"
            "  --model-dir <path>   Directory with manifest.json\n"
            "  --prompt <text>      Input text prompt\n"
            "  --max-tokens N       Max tokens to generate (default: 50)\n"
            "  --temperature T      Sampling temperature (default: 0.8)\n"
            "  --mock               Create and run a mock model for testing\n"
            "  --mock-hidden N      Mock hidden size (default: 256)\n"
            "  --mock-intermediate N Mock intermediate size (default: 1024)\n"
            "  --mock-layers N      Mock num layers (default: 2)\n"
            "  --mock-vocab N       Mock vocab size (default: 100)\n"
            "  --mock-seq-len N     Mock max seq len (default: 64)\n"
            "  --temp-dir <path>    Directory for temp files (default: /tmp)\n"
            "  --help               Show this help\n"
            "  --version            Show version\n"
            "\n"
            "Examples:\n"
            "  %s --mock --prompt \"hello world\"\n"
            "  %s --model model.tbm --prompt \"Once upon a time\"\n",
            prog, prog, prog);
}

static void print_version() {
    fprintf(stderr, "tensorbit-run v1.0.0\n"
                     "Part of the Tensorbit Labs P-D-Q pipeline\n");
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char** argv) {
    /* Default options */
    std::string model_path;
    std::string model_dir;
    std::string prompt = "hello";
    int         max_tokens = 50;
    float       temperature = 0.8f;
    bool        mock_mode = false;
    int         mock_hidden = 256;
    int         mock_intermediate = 1024;
    int         mock_layers = 2;
    int         mock_vocab = 100;
    int         mock_seq_len = 64;
    std::string temp_dir = ".";

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--model-dir" && i + 1 < argc)
            model_dir = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)
            prompt = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (arg == "--mock")
            mock_mode = true;
        else if (arg == "--mock-hidden" && i + 1 < argc)
            mock_hidden = atoi(argv[++i]);
        else if (arg == "--mock-intermediate" && i + 1 < argc)
            mock_intermediate = atoi(argv[++i]);
        else if (arg == "--mock-layers" && i + 1 < argc)
            mock_layers = atoi(argv[++i]);
        else if (arg == "--mock-vocab" && i + 1 < argc)
            mock_vocab = atoi(argv[++i]);
        else if (arg == "--mock-seq-len" && i + 1 < argc)
            mock_seq_len = atoi(argv[++i]);
        else if (arg == "--temp-dir" && i + 1 < argc)
            temp_dir = argv[++i];
        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--version") {
            print_version();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Initialize backends */
    BackendRegistry::instance().init();

    /* Load model */
    Model model;
    bool  owns_temp_file = false;
    std::string temp_path;

    if (mock_mode) {
        temp_path = temp_dir + "/mock_model.tbm";
        TB_LOG_INFO("creating mock model: hidden={}, layers={}, vocab={}", mock_hidden, mock_layers,
                    mock_vocab);

        int ret = create_mock_tbm(temp_path.c_str(), mock_hidden, mock_intermediate, mock_layers,
                                   mock_vocab, mock_seq_len);
        if (ret != TB_OK) {
            auto err_str = tb_error_string(ret);
            TB_LOG_ERROR("failed to create mock model: {}", err_str);
            return 1;
        }
        owns_temp_file = true;

        {
            auto tmp = temp_path.c_str();
            fprintf(stderr, "Mock model created at %s\n", tmp);
        }

        auto result = Model::load_tbm(temp_path);
        if (!result) {
            auto err_code = (int)result.error();
            TB_LOG_ERROR("failed to load mock model: {}", err_code);
            return 1;
        }
        model = std::move(*result);
    } else if (!model_path.empty()) {
        auto result = Model::load_tbm(model_path);
        if (!result) {
            auto mp = model_path.c_str();
            TB_LOG_ERROR("failed to load model from {}", mp);
            return 1;
        }
        model = std::move(*result);
    } else if (!model_dir.empty()) {
        auto result = Model::load_dir(model_dir);
        if (!result) {
            auto md = model_dir.c_str();
            TB_LOG_ERROR("failed to load model from {}", md);
            return 1;
        }
        model = std::move(*result);
    } else {
        fprintf(stderr, "Error: specify --model, --model-dir, or --mock\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Print model info */
    auto cfg = model.config();
    fprintf(stderr, "Model loaded:\n");
    fprintf(stderr, "  Architecture: %s\n", cfg.architecture.c_str());
    fprintf(stderr, "  Hidden size:  %d\n", cfg.hidden_size);
    fprintf(stderr, "  Layers:       %d\n", cfg.num_layers);
    fprintf(stderr, "  Heads:        %d (%d KV)\n", cfg.num_heads, cfg.num_kv_heads);
    fprintf(stderr, "  Head dim:     %d\n", cfg.head_dim);
    fprintf(stderr, "  Intermediate: %d\n", cfg.intermediate_size);
    fprintf(stderr, "  Vocab size:   %d\n", cfg.vocab_size);
    fprintf(stderr, "  Max seq len:  %d\n", cfg.max_seq_len);
    fprintf(stderr, "  Total layers: %d\n", model.num_layers());

    /* Setup tokenizer */
    SimpleTokenizer tokenizer;
    tokenizer.init_default();

    /* Setup runner */
    TransformerRunner runner(model);
    runner.init_state();

    /* Encode prompt */
    std::vector<int> token_ids = tokenizer.encode(prompt);

    fprintf(stderr, "\nPrompt: \"%s\"\n", prompt.c_str());
    fprintf(stderr, "Tokens:  ");
    for (int id : token_ids) fprintf(stderr, "%d ", id);
    fprintf(stderr, "\n\nGenerated: ");

    /* Forward pass for each prompt token */
    for (size_t i = 0; i < token_ids.size(); i++) {
        const float* logits = runner.forward(token_ids[i]);
        if (!logits) {
            auto tok_idx = i;
            TB_LOG_ERROR("forward pass failed at token {}", tok_idx);
            break;
        }
    }

    /* Generate new tokens */
    for (int t = 0; t < max_tokens; t++) {
        if (runner.vocab_size() <= 0) break;

        const float* logits = runner.logits();
        if (!logits) break;

        /* Re-run forward for subsequent tokens (use last generated token as input) */
        if (t > 0) {
            /* Already ran forward for previous generated token above */
        }

        int next_token = sample_temperature(logits, runner.vocab_size(), temperature);
        std::string token_str = tokenizer.decode(next_token);
        fprintf(stdout, "%s", token_str.c_str());
        fflush(stdout);

        /* Feed back into model for next iteration */
        runner.forward(next_token);
    }

    fprintf(stdout, "\n");

    /* Cleanup mock temp file */
    if (owns_temp_file) {
        remove(temp_path.c_str());
    }

    return 0;
}
