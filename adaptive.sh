# python main_adaptive.py --shape box --pretrain_only

python main_adaptive.py \
    --file data/stanford-bunny.ply \
    --subdiv_threshold 0.3 \
    --subdiv_max_depth 5 \
    --lam_svd 0.1 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000