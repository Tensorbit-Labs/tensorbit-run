#include "tensorbit/run/ops.h"
#include <string.h>

int tb_cpu_embedding_f32(const int* TB_RESTRICT token_ids, int n_tokens, int embedding_dim,
                          const float* TB_RESTRICT table, float* TB_RESTRICT y) {
    if (!token_ids || !table || !y) return TB_ERR_INVALID_ARG;

    for (int t = 0; t < n_tokens; t++) {
        int tid = token_ids[t];
        memcpy(y + t * embedding_dim, table + tid * embedding_dim, sizeof(float) * embedding_dim);
    }

    return TB_OK;
}
