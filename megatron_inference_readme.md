## Login to HuggingFace to access models

huggingface-cli login
hf token: <hftoken>

## Download model checkpoints

- Make a directory for the model in the save path for both the hf and megatron versions.

```bash

python download_hf_checkpoints.py \
    --repo_id "meta-llama/Meta-Llama-3-8B" \
    --download_path "/mnt/lustre/gaia/sshrestha/hf_models/meta-llama-3-8B" \
    --allow_patterns "config.json" "generation_config.json" "*.safetensors" "model.safetensors.index.json" "special_tokens_map.json" "tokenizer.json" "tokenizer.model" "tokenizer_config.json"
```

## Convert HF checkpoints to Megatron format

```bash
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --load-dir /mnt/lustre/gaia/sshrestha/hf_models/meta-llama-3-8B \
    --model-size llama3 \
    --checkpoint-type hf \
    --saver core \
    --save-dir /mnt/lustre/gaia/sshrestha/megatron_models/meta-llama-3-8B \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --bf16 \
    --tokenizer-model meta-llama/Meta-Llama-3-8B
```

## Run inference

### Validate checkpoints

```bash
examples/inference/llama_mistral/run_text_generation_llama3.sh /mnt/lustre/gaia/sshrestha/megatron_models/meta-llama-3-8B /mnt/lustre/gaia/sshrestha/hf_models/meta-llama-3-8B 
```

Once running, query the server with 

```bash
curl 'http://<TEXT_GENERATION_SERVER_IP>:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["The tallest mountain in the world is the" ], "tokens_to_generate":100, "top_k":1}'
```

curl 'http://10.230.196.111:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["The tallest mountain in the world is the" ], "tokens_to_generate":100, "top_k":1}'

Run performance script

required for tensor parallel inference 

export CUDA_DEVICE_MAX_CONNECTIONS=1
```bash
torchrun --nproc_per_node 1 --nnodes 1 tools/run_inference_performance_test.py \
    --use-checkpoint-args --use-flash-attn \
    --disable-bias-linear --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-8B  \
    --transformer-impl transformer_engine --te-rng-tracker \
    --normalization RMSNorm \
    --group-query-attention --num-query-groups 8 \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 --rotary-base 500000 \
    --use-rotary-position-embeddings \
    --swiglu \
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 1  \
    --num-layers 32  \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --load /mnt/lustre/gaia/sshrestha/megatron_models/meta-llama-3-8B \
    --num-attention-heads 32  \
    --max-position-embeddings 8192  \
    --fp16  \
    --micro-batch-size 1  \
    --num-tokens-to-generate 100 \
    --seq-length 8192 \
    --prompts "The tallest mountain in the world is " \
    --enable-cuda-graph \
    --engine-type dynamic \
```
## General model inference with no checkpoint for experiments


## Llama-3-8b 

```bash
torchrun --nproc_per_node 4 --nnodes 1 tools/run_inference_performance_test.py     --use-flash-attn     --disable-bias-linear --tokenizer-type HuggingFaceTokenizer     --tokenizer-model meta-llama/Meta-Llama-3-8B      --transformer-impl transformer_engine --te-rng-tracker     --normalization RMSNorm     --group-query-attention --num-query-groups 8     --attention-dropout 0.0 --hidden-dropout 0.0     --untie-embeddings-and-output-weights     --position-embedding-type rope     --rotary-percent 1.0 --rotary-base 500000     --use-rotary-position-embeddings     --swiglu     --tensor-model-parallel-size 1      --pipeline-model-parallel-size 4      --num-layers 32      --hidden-size 4096      --ffn-hidden-size 14336     --num-attention-heads 32      --max-position-embeddings 8192      --fp16      --micro-batch-size 1      --num-tokens-to-generate 100     --seq-length 8192     --inference-max-requests 256     --max-batch-size 256     --num-input-tokens 1024 --benchmark-profile      --enable-cuda-graph 
```


## Llama-3-70b
```bash
torchrun --nproc_per_node 2 --nnodes 1 tools/run_inference_performance_test.py \
    --tensor-model-parallel-size 2  \
    --pipeline-model-parallel-size 1  \
    --use-flash-attn \
    --disable-bias-linear --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Meta-Llama-3-70B  \
    --transformer-impl transformer_engine --te-rng-tracker \
    --normalization RMSNorm \
    --group-query-attention --num-query-groups 8 \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 --rotary-base 500000 \
    --use-rotary-position-embeddings \
    --swiglu \
    --num-layers 80  \
    --hidden-size 8192  \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64  \
    --max-position-embeddings 8192  \
    --fp16  \
    --micro-batch-size 1  \
    --num-tokens-to-generate 100 \
    --seq-length 8192 \
    --inference-max-requests 8 \
    --max-batch-size 8 \
    --num-input-tokens 1024 --benchmark-profile \
    --enable-cuda-graph 
```

## Llama 4 scout

```bash
torchrun --nproc_per_node 8 --nnodes 1 tools/run_inference_performance_test.py \
    --tensor-model-parallel-size 4  \
    --pipeline-model-parallel-size 2  \
    --use-flash-attn \
    --flash-decode \
    --tokenizer-model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 48 \
    --hidden-size 5120 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --num-experts 16 \
    --moe-ffn-hidden-size 8192 \
    --moe-router-score-function sigmoid \
    --moe-router-topk 1 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-shared-expert-intermediate-size 8192 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-expert-capacity-factor 1.25 \
    --moe-pad-expert-input-to-capacity \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-interleaved \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --rope-scaling-factor 8.0 \
    --use-rope-scaling \
    --no-bias-swiglu-fusion \
    --qk-l2-norm \
    --moe-apply-probs-on-input \
    --moe-router-dtype fp64 \
    --num-tokens-to-generate 100 \
    --inference-max-requests 8 \
    --max-batch-size 8 \
    --num-input-tokens 1024 --benchmark-profile \
    --transformer-impl transformer_engine --te-rng-tracker \
    --enable-cuda-graph 
```

# Grok inference

```bash
torchrun --nproc_per_node 8 --nnodes 1 tools/run_inference_performance_test.py \
    --tensor-model-parallel-size 8  \
    --pipeline-model-parallel-size 1  \
    --expert-tensor-parallel-size 1 \
    --use-flash-attn \
    --flash-decode \
    --tokenizer-model meta-llama/Meta-Llama-3-70B \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 64 \
    --hidden-size 6144 \
    --ffn-hidden-size 32768  \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --num-experts 8 \
    --moe-ffn-hidden-size 8192 \
    --moe-router-score-function sigmoid \
    --moe-router-topk 2 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-expert-capacity-factor 1.25 \
    --moe-pad-expert-input-to-capacity \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-interleaved \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --rope-scaling-factor 8.0 \
    --use-rope-scaling \
    --no-bias-swiglu-fusion \
    --qk-l2-norm \
    --moe-router-dtype fp64 \
    --num-tokens-to-generate 100 \
    --inference-max-requests 8 \
    --max-batch-size 8 \
    --num-input-tokens 1024 --benchmark-profile \
    --transformer-impl transformer_engine --te-rng-tracker \
    --enable-cuda-graph 
```