set -e
GPU_IDS=${1:-"0,1,2,3"}

MODELS=(
    ""
)



MAX_NEW_TOKENS=8192
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=-1
MAX_MODEL_LEN=16384
LIMIT_MM=10

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "[INFO] Using ${NUM_GPUS} GPUs for data parallelism: GPU=${GPU_IDS}"
echo "[INFO] Model count: ${#MODELS[@]}, dataset count: ${#INPUTS[@]}"
echo "========================================"

get_model_name() {
    basename "$1"
}

get_input_name() {
    local fname
    fname=$(basename "$1")
    echo "${fname%.*}"
}

run_single() {
    local model=$1
    local input=$2
    local output=$3

    if [ ${NUM_GPUS} -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=${GPUS[0]} python infer.py \
          --model "${model}" \
          --input "${input}" \
          --output "${output}" \
          --tensor-parallel 1 \
          --enable-chunked-prefill \
          --max-new-tokens ${MAX_NEW_TOKENS} \
          --temperature ${TEMPERATURE} \
          --top-p ${TOP_P} \
          --top-k ${TOP_K} \
          --max-model-len ${MAX_MODEL_LEN} \
          --limit-mm-per-prompt ${LIMIT_MM}
    else
        local shard_dir="output/.shards_$$"
        mkdir -p "${shard_dir}"
        local PIDS=()

        for i in "${!GPUS[@]}"; do
            local GPU=${GPUS[$i]}
            local SHARD_OUTPUT="${shard_dir}/shard_${i}.jsonl"
            echo "  [INFO] Launching GPU=${GPU} for shard ${i}/${NUM_GPUS}"
            CUDA_VISIBLE_DEVICES=${GPU} python infer.py \
              --model "${model}" \
              --input "${input}" \
              --output "${SHARD_OUTPUT}" \
              --tensor-parallel 1 \
              --enable-chunked-prefill \
              --max-new-tokens ${MAX_NEW_TOKENS} \
              --temperature ${TEMPERATURE} \
              --top-p ${TOP_P} \
              --top-k ${TOP_K} \
              --max-model-len ${MAX_MODEL_LEN} \
              --limit-mm-per-prompt ${LIMIT_MM} \
              --shard-id ${i} \
              --num-shards ${NUM_GPUS} &
            PIDS+=($!)
        done

        echo "  [INFO] Waiting for ${NUM_GPUS} processes to finish..."
        for pid in "${PIDS[@]}"; do
            wait $pid
        done

        echo "  [INFO] Merging shard results into ${output}"
        cat ${shard_dir}/shard_*.jsonl > "${output}"
        rm -rf "${shard_dir}"
    fi
}

mkdir -p output
TOTAL=$((${#MODELS[@]} * ${#INPUTS[@]}))
COUNT=0

for model in "${MODELS[@]}"; do
    model_name=$(get_model_name "${model}")
    for input in "${INPUTS[@]}"; do
        input_name=$(get_input_name "${input}")
        COUNT=$((COUNT + 1))

        output="output/${input_name}_${model_name}.jsonl"

        echo ""
        echo "[${COUNT}/${TOTAL}] Model: ${model_name} | Dataset: ${input_name}"
        echo "[${COUNT}/${TOTAL}] Output: ${output}"

        if [ -f "${output}" ]; then
            echo "  [SKIP] Output file exists, skipping"
            continue
        fi

        run_single "${model}" "${input}" "${output}"
        echo "[DONE] ${output}"
    done
done