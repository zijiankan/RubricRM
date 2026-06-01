set -x

project_name='T2I-RM-GRPO'
exp_name='T2I-RM-Qwen3_5_9B'
gen_tp=1
sp_size=1
NGPUS=${NGPUS:-8}
ENGINE=${1:-vllm}
VERL_DIR="./"
CACHE_DIR=${CACHE_DIR:-"./cache"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${CACHE_DIR}/ray"}
DATA_DIR=${DATA_DIR:-""} #change to your data dir
MODEL_PATH=${MODEL_PATH:-""} #change to your mdoel path
CKPTS_DIR=${CKPTS_DIR:-""} #change to yourcheckpoint path
# Mode: v1 (full generation) or v2 (rubric-guided continuation)
TRAIN_MODE=${TRAIN_MODE:-v2}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
VAL_DIR=${VAL_DIR:-""} #change to your val data path
VAL_FILE=${VAL_FILE:-"${VAL_DIR}/val_gen.parquet"}
REWARD_FILE=${REWARD_FILE:-"${VERL_DIR}/examples/reward_function/t2i_rm_reward.py"}

# if Qwen3.5:
TEMPLATE_FLAGS="+data.apply_chat_template_kwargs.enable_thinking=false"
if [ "${TRAIN_MODE}" = "v2" ]; then
    CONTINUE_FLAG="${TEMPLATE_FLAGS} +data.apply_chat_template_kwargs.continue_final_message=true"
    MAX_RESP_LEN=2048
else
    CONTINUE_FLAG="${TEMPLATE_FLAGS}"
    MAX_RESP_LEN=8192
fi
WORKING_DIR=${WORKING_DIR:-"${VERL_DIR}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${VERL_DIR}/verl/trainer/runtime_env.yaml"}

start_time=$(date +%Y%m%d)_$(date +%H%M%S)

export HF_HOME="${CACHE_DIR}/huggingface"
export TMPDIR="${CACHE_DIR}/tmp"
mkdir -p "${CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"

export VLLM_BATCH_INVARIANT=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Reward JSON log directory (auto-created); set sample rate to avoid bloat
export T2I_REWARD_LOG_DIR="${VERL_DIR}/logs/reward/${TRAIN_MODE}_${start_time}"
export T2I_REWARD_LOG_SAMPLE_RATE=1

mkdir -p logs
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.val_batch_size=256 \
    data.train_batch_size=64 \
    data.max_prompt_length=16384 \
    data.max_response_length=${MAX_RESP_LEN} \
    +data.cache_dir=${CACHE_DIR}/verl/rlhf \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.num_workers=${NGPUS} \
    +actor_rollout_ref.rollout.limit_images=8 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.optim.clip_grad=0.5 \
    +algorithm.grpo_reward_saturation_std=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.balance_batch=True \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=8192 \
    trainer.total_epochs=1 ${CONTINUE_FLAG} $@ 2>&1 | tee logs/${start_time}.log