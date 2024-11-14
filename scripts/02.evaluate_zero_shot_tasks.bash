cd "$(dirname "$0")/.."

run_evaluation() {
    local model_id=$1
    local target_sparsity_config=$2

    local work_folder=./results/scap/$model_id/$target_sparsity_config/
    local calibrated_thresholds_json_path=$work_folder/calibrated_thresholds.json
    local evaluation_results_json_path=$work_folder/evaluation_results.json

    mkdir -p $work_folder

    echo "[$(date)] Evaluation started for $model_id : $target_sparsity_config"
    python -u evaluate.py \
        --model_id $model_id \
        --model_load_dtype float16 \
        --calibrated_thresholds_json_path $calibrated_thresholds_json_path \
        --evaluation_tasks wikitext,winogrande,piqa,sciq,hellaswag,boolq,arc_easy,arc_challenge \
        --evaluation_results_json_path $evaluation_results_json_path 2>&1 | tee -a $work_folder/evaluation.log
    echo "[$(date)] Evaluation finished for $model_id : $target_sparsity_config"
}

run_evaluation "meta-llama/Llama-2-7b-hf" "up,zero,0.35,gate,zero,0.35,down,zero,0.55"
run_evaluation "mistralai/Mistral-7B-v0.1" "up,zero,0.3,gate,zero,0.3,down,zero,0.7"
run_evaluation "mosaicml/mpt-7b" "down,kde,0.5"
run_evaluation "tiiuae/falcon-7b" "down,median,0.5"
wait
