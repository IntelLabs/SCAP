cd "$(dirname "$0")/.."

run_calibration() {
    local model_id=$1
    local target_sparsity_config=$2

    local work_folder=./results/scap/$model_id/$target_sparsity_config/
    local calibrated_thresholds_json_path=$work_folder/calibrated_thresholds.json
    local evaluation_results_json_path=$work_folder/evaluation_results.json

    mkdir -p $work_folder

    echo "[$(date)] Calibration started for $model_id : $target_sparsity_config"
    python -u calibrate.py \
        --model_id $model_id \
        --model_load_dtype float32 \
        --target_sparsity_config $target_sparsity_config \
        --calibrated_thresholds_json_path $calibrated_thresholds_json_path 2>&1 | tee -a $work_folder/calibration.log
    echo "[$(date)] Calibration finished for $model_id : $target_sparsity_config"
}

run_calibration "meta-llama/Llama-2-7b-hf" "up,zero,0.35,gate,zero,0.35,down,zero,0.55"
run_calibration "mistralai/Mistral-7B-v0.1" "up,zero,0.3,gate,zero,0.3,down,zero,0.7"
run_calibration "mosaicml/mpt-7b" "down,kde,0.5"
run_calibration "tiiuae/falcon-7b" "down,median,0.5"
wait
