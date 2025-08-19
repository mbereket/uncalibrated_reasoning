# generate predictions from trained models
uv run eval.py --experiment-name final_grpo \
                  --step 180 \
                  --split test

uv run eval.py --experiment-name final_grpo_no_std \
                  --step 180 \
                  --split test

uv run eval.py --experiment-name final_rloo \
                  --step 180 \
                  --split test

uv run eval.py --experiment-name final_ppo \
                  --step 180 \
                  --split test

# generate zero-shot predictions
uv run eval.py --pretrained-model-name Qwen3-4B --split test
