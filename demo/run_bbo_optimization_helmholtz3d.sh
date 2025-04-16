#!/bin/bash

# Set error handling
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/helmholtz_optimization_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run optimization algorithm
run_algorithm() {
    local algorithm=$1
    local wave_number=$2
    log_message "Starting optimization with $algorithm algorithm (k=$wave_number)..."
    
    # Run the algorithm and capture both stdout and stderr
    python demo_bbo_optimization_helmholtz.py \
        --algorithm "$algorithm" \
        --nc 1000 \
        --nb 200 \
        --nc-test 20 \
        --seed 42 \
        --epochs 10000 \
        --n-trials 150 \
        --population-size 32 \
        --wave-number "$wave_number" \
        --domain-size 1.0 2>&1 | tee -a "$LOG_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        log_message "Successfully completed $algorithm optimization (k=$wave_number)"
    else
        log_message "Error: Failed to complete $algorithm optimization (k=$wave_number)"
        return 1
    fi
}

# Main execution
log_message "Starting Helmholtz optimization process for all algorithms"

# List of algorithms to run
ALGORITHMS=("whales")

# Wave numbers to test
WAVE_NUMBERS=(1.0)

# Run each algorithm with each wave number
for algorithm in "${ALGORITHMS[@]}"; do
    for wave_number in "${WAVE_NUMBERS[@]}"; do
        log_message "----------------------------------------"
        log_message "Running $algorithm algorithm with k=$wave_number"
        log_message "----------------------------------------"
        
        # Run the algorithm and check for errors
        if ! run_algorithm "$algorithm" "$wave_number"; then
            log_message "Error: Stopping optimization process due to failure in $algorithm with k=$wave_number"
            exit 1
        fi
        
        # Add a small delay between runs
        sleep 5
    done
done

log_message "----------------------------------------"
log_message "All algorithms completed successfully"
log_message "----------------------------------------" 