#!/bin/bash

# Set error handling
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/flow_mixing_optimization_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run optimization algorithm
run_algorithm() {
    local algorithm=$1
    log_message "Starting optimization with $algorithm algorithm..."
    
    # Run the algorithm and capture both stdout and stderr
    python demo_bbo_optimization_flowmixing3d.py \
        --algorithm "$algorithm" \
        --nc 1000 \
        --ni 100 \
        --nb 100 \
        --nc-test 50 \
        --seed 42 \
        --epochs 10000 \
        --n-trials 150 \
        --population-size 32 \
        --a 1.0 \
        --b 1.0 2>&1 | tee -a "$LOG_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        log_message "Successfully completed $algorithm optimization"
    else
        log_message "Error: Failed to complete $algorithm optimization"
        return 1
    fi
}

# Main execution
log_message "Starting Flow Mixing 3D optimization process for all algorithms"

# List of algorithms to run
ALGORITHMS=("jade" "lshade" "nelder_mead" "pso" "grey_wolf" "whales")

# Run each algorithm
for algorithm in "${ALGORITHMS[@]}"; do
    log_message "----------------------------------------"
    log_message "Running $algorithm algorithm"
    log_message "----------------------------------------"
    
    # Run the algorithm and check for errors
    if ! run_algorithm "$algorithm"; then
        log_message "Error: Stopping optimization process due to failure in $algorithm"
        exit 1
    fi
    
    # Add a small delay between algorithms
    sleep 5
done

log_message "----------------------------------------"
log_message "All algorithms completed successfully"
log_message "----------------------------------------" 