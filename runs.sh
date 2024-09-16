# transformer for 400ctx no patch
run train_sim_WH_skip --out-file ckpt_400ctx_nopatch --cuda-device cuda:1 --seq-len-ctx 400 --seq-len-skip 400 --seq-len-n-in 10 --log-wandb

# transformer for 800ctx RNN patch
run train_sim_WH_skip --out-file ckpt_400ctx_nopatch --cuda-device cuda:1 --seq-len-ctx 800 --seq-len-skip 400 --seq-len-n-in 10 --log-wandb

# transformer for 16000ctx RNN patch
run train_sim_WH_skip --out-file ckpt_400ctx_nopatch --cuda-device cuda:1 --seq-len-ctx 16_000 --seq-len-skip 400 --seq-len-n-in 10 --log-wandb

# transformer for 40000ctx RNN patch
run train_sim_WH_skip --out-file ckpt_400ctx_nopatch --cuda-device cuda:1 --seq-len-ctx 40_000 --seq-len-skip 400 --seq-len-n-in 10 --log-wandb
