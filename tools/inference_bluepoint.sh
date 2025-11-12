# Define paths
TEST_LIST_INIT="data/ForAINetV2/meta_data/test_list_initial.txt"
TEST_LIST="data/ForAINetV2/meta_data/test_list.txt"
TEST_DATA_DIR="data/ForAINetV2/test_data"
WORK_DIR="/workspace"
CONFIG_FILE="$WORK_DIR/configs/oneformer3d_qs_radius16_qp300_2many.py"
MODEL_PATH="$WORK_DIR/work_dirs/clean_forestformer/epoch_3000_fix.pth"
ITERATIONS=2  # Default number of iterations
####BLUEPOINTS_DIR="$WORK_DIR/work_dirs/bluepoint_th04fixed_03_priority_test"
BLUEPOINTS_DIR="$WORK_DIR/work_dirs/V3"

find data/ForAINetV2/forainetv2_instance_data -type f -name "*bluepoints*" -delete

# Iterate through all test files listed in test_list_initial.txt
while IFS= read -r scan_name || [ -n "$scan_name" ]; do
    echo "Processing: $scan_name"
    
    iteration=1
    current_scan_name="$scan_name"
    
    while [ "$iteration" -le "$ITERATIONS" ]; do
        echo "Iteration $iteration for $current_scan_name"
        
        # Update test_list.txt to point to the current scan_name
        echo "$current_scan_name" > "$TEST_LIST"
        
        # Navigate to the data directory and run the data loading script
        cd "$WORK_DIR/data/ForAINetV2" || exit
        python batch_load_ForAINetV2_data.py --test_scan_names_file meta_data/test_list.txt
        cd "$WORK_DIR" || exit
        
        # Run the data processing script
        python tools/create_data_forainetv2.py forainetv2

        # modify score_th
        #score_th=$(echo "scale=2; 0.2 + ($iteration - 1) * 0.1" | bc)
        score_th=0.4

        # modify CONFIG_FILE
        sed -i "s/score_th = [0-9.]\+/score_th = 0$score_th/g" "$CONFIG_FILE"
        
        # Run the testing script
        CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CONFIG_FILE" "$MODEL_PATH"
        
        # Generated new prediction file
        new_pre_FILE="${scan_name}_${iteration}.ply"
        new_pre_PATH="$BLUEPOINTS_DIR/$new_pre_FILE"
        
        if [ ! -f "$new_pre_PATH" ]; then
            echo "No more new prediction file found. Ending iterations for $scan_name."
            break
        fi
        
        # Generated bluepoints file
        BLUEPOINTS_FILE="${scan_name}_bluepoints_${iteration}.ply"
        BLUEPOINTS_PATH="$BLUEPOINTS_DIR/$BLUEPOINTS_FILE"
        
        # Check if the bluepoints file exists
        if [ ! -f "$BLUEPOINTS_PATH" ]; then
            echo "No more bluepoints file found. Ending iterations for $scan_name."
            break
        fi
        
        # Copy the bluepoints file to the test_data directory
        cp "$BLUEPOINTS_PATH" "$TEST_DATA_DIR/"
        
        # Update current scan_name to process the next round of bluepoints
        current_scan_name="${BLUEPOINTS_FILE%.ply}"
        ((iteration++))
    done

    # Merge prediction files for the scan 
    echo "Merging results for $scan_name" 
    python tools/merge_prediction.py "$scan_name" "$BLUEPOINTS_DIR" "$ITERATIONS" 
    
    echo "Finished processing $scan_name. Moving to next."

done < "$TEST_LIST_INIT"

# Evaluate each round's results
for ((i=1; i<=ITERATIONS; i++)); do
    ROUND_DIR="$BLUEPOINTS_DIR/round_$i"
    echo "Evaluating results in: $ROUND_DIR"
    python tools/final_eval.py "$ROUND_DIR"
done

# Evaluate results after noise removal (assuming any number suffix)
for ((i=1; i<=ITERATIONS; i++)); do
    for ROUND_DIR in "$BLUEPOINTS_DIR"/round_"$i"_after_remove_noise_*; do
        if [ -d "$ROUND_DIR" ]; then
            echo "Evaluating results in: $ROUND_DIR"
            python tools/final_eval.py "$ROUND_DIR"
        fi
    done
done

echo "All test cases processed."
