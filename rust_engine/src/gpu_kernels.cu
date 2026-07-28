extern "C" __global__ void backtest_kernel(
    const unsigned int* bitsets, // [70][N_U32] - precombined bitsets
    const float* prices,               // [N]
    const float* atrs,                 // [N]
    const float* btc_vols,             // [N]
    const float* genome_params,        // [N_GENOMES][5] (vol_idx, body_idx, ema_f, tp_rr, sl_mult)
    int n_candles,
    int n_u32,                         // ceil(n_candles / 32)
    int n_genomes,
    float* out_fitness                 // [N_GENOMES]
) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= n_genomes) return;

    // Local copy of params
    float p_vol_idx = genome_params[g_idx * 5 + 0];
    float p_body_idx = genome_params[g_idx * 5 + 1];
    float p_ema_f = genome_params[g_idx * 5 + 2];
    float tp_rr = genome_params[g_idx * 5 + 3];
    float sl_mult = genome_params[g_idx * 5 + 4];

    // Identify which discrete bitset index to use: 70 total
    // vol(7) * body(5) * ema(2)
    int b_idx = ((int)p_vol_idx * 10) + ((int)p_body_idx * 2) + (int)p_ema_f;
    const unsigned int* my_bitset = &bitsets[b_idx * n_u32];

    float equity = 0.0f;
    int trades = 0;
    int wins = 0;
    float gross_profit = 0.0f;
    float gross_loss = 0.0f;
    float downside_sq = 0.0f;
    float peak_equity = 0.0f;
    float max_dd = 0.0f;

    int cooldown_until = -1;

    // SYNCED: Slippage constant (0.04% per side = 0.08% round-trip)
    const float SLIPPAGE_PCT = 0.0004f;

    // Simulation loop
    for (int i = 20; i < n_candles - 50; i++) {
        if (i < cooldown_until) continue;

        // Check entry signal (bitset)
        bool signal = (my_bitset[i / 32] >> (i % 32)) & 1;
        if (signal) {
            float entry_price = prices[i];
            float atr = atrs[i];
            if (atr <= 0) continue;

            float sl_dist = atr * sl_mult;
            float tp_dist = sl_dist * tp_rr;
            
            float sl_price = entry_price - sl_dist;
            float tp_price = entry_price + tp_dist;

            // Simple forward scan for TP/SL (max 100 bars)
            bool closed = false;
            float pnl_r = 0.0f;
            for (int j = i + 1; j < i + 100 && j < n_candles; j++) {
                if (prices[j] <= sl_price) {
                    pnl_r = -1.0f;
                    closed = true;
                    cooldown_until = j + 12; // 1h cooldown
                    break;
                }
                if (prices[j] >= tp_price) {
                    pnl_r = tp_rr;
                    closed = true;
                    cooldown_until = j + 12;
                    break;
                }
            }

            if (closed) {
                // SYNCED: Apply slippage (0.08% round-trip reduces PnL)
                float slip_cost = entry_price * SLIPPAGE_PCT * 2.0f; // entry + exit
                float slip_r = slip_cost / sl_dist; // convert to R-multiples
                pnl_r -= slip_r;

                // SYNCED: Recency weighting (matches CPU: weight = 0.3 + 0.7 * ratio)
                float ratio = (float)i / (float)n_candles;
                float weight = 0.3f + 0.7f * ratio;
                if (weight > 1.0f) weight = 1.0f;

                // SYNCED: Stress penalty (matches CPU exactly)
                float vol = btc_vols[i];
                if (vol > 1.5f) {
                    if (pnl_r > 0) pnl_r *= 0.5f;   // Lucky profit penalty
                    else pnl_r *= 1.3f;              // Stress amplification
                }

                // Apply recency weight
                pnl_r *= weight;

                equity += pnl_r;
                trades++;
                if (pnl_r > 0) {
                    wins++;
                    gross_profit += pnl_r;
                } else {
                    gross_loss += fabsf(pnl_r);
                    downside_sq += pnl_r * pnl_r;
                }

                if (equity > peak_equity) peak_equity = equity;
                float dd = peak_equity - equity;
                if (dd > max_dd) max_dd = dd;
            }
        }
    }

    // SYNCED: Min trades threshold matches CPU (30)
    if (trades < 30) {
        out_fitness[g_idx] = -100.0f;
        return;
    }

    float avg_pnl = equity / (float)trades;
    float downside_dev = sqrtf(downside_sq / (float)trades);
    // SYNCED: Sortino else case matches CPU (*10.0)
    float sortino = (downside_dev > 0) ? (avg_pnl / downside_dev) : (avg_pnl * 10.0f);
    
    // SYNCED: PF cap matches CPU (10.0)
    float pf = (gross_loss > 0) ? (gross_profit / gross_loss) : 10.0f;
    float log_trades = logf((float)trades);
    if (log_trades < 1.0f) log_trades = 1.0f;
    float dd_penalty = (max_dd > 3.0f) ? (1.0f - fminf(0.9f, (max_dd - 3.0f) * 0.1f)) : 1.0f;

    out_fitness[g_idx] = sortino * log_trades * pf * dd_penalty;
}
