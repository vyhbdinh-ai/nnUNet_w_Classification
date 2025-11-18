import time
import subprocess
import pandas as pd
from pathlib import Path
import argparse
import sys
import os

def run_inference(config_name, enable_tta, step_size, input_folder, output_folder, model_folder):    
    # Create output folder for each configuration
    config_output_folder = Path(output_folder) / config_name
    config_output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "multitask_predict.py",
        "--input_folder", input_folder,
        "--output_folder", str(config_output_folder),
        "--model_folder", model_folder,
        "--step_size", str(step_size) ]
    
    if enable_tta:
        cmd.append("--enable_tta")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        execution_time = end_time - start_time
                
        # Check if outputs were created
        seg_files = list(config_output_folder.glob("*.nii.gz"))
        csv_file = config_output_folder / "subtype_results.csv"
        
        if len(seg_files) > 0 and csv_file.exists():
            print(f"Outputs: {len(seg_files)} segmentations + classification CSV")
        
        return execution_time, True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Failed after {execution_time:.2f} seconds")
        print(f"Error: {e.stderr}")
        return execution_time, False

def compare_speeds(args):
    configurations = [
        {
            "name": "baseline",
            "enable_tta": True,
            "step_size": 0.5,
            "description": "Default settings (TTA enabled, step_size=0.5)"
        },
        {
            "name": "optimized", 
            "enable_tta": False,
            "step_size": 0.7,
            "description": "Optimized (TTA disabled, step_size=0.7)"
        }]
    
    results = []
    for config in configurations:
        execution_time, success = run_inference(
            config["name"],
            config["enable_tta"],
            config["step_size"],
            args.input_folder,
            args.output_folder,
            args.model_folder)
        results.append({
            "Configuration": config["name"],
            "Description": config["description"],
            "TTA": "Enabled" if config["enable_tta"] else "Disabled",
            "Step Size": config["step_size"],
            "Time (seconds)": round(execution_time, 2),
            "Status": "Success" if success else "Failed"})
    
    baseline_time = results[0]["Time (seconds)"]

    for i, result in enumerate(results[1:], 1):
        optimized_time = result["Time (seconds)"]
        speedup = baseline_time / optimized_time
        improvement_pct = ((baseline_time - optimized_time) / baseline_time) * 100
        results[i]["Improvement %"] = round(improvement_pct, 1)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    if len(results) >= 2 and all(r["Status"] == "Success" for r in results):        
        baseline = results[0]
        optimized = results[1]
        num_cases = len(results)/2
        print(f"Baseline (TTA enabled, step_size=0.5):")
        print(f"Time: {baseline['Time (seconds)']} seconds")
        print(f"Time per case: {baseline['Time (seconds)']/num_cases} seconds")
        print(f"\nOptimized (TTA disabled, step_size=0.75):")
        print(f"Time: {optimized['Time (seconds)']} seconds")
        print(f"Improvement: {optimized.get('Improvement %', 'N/A')}% faster")
    
    # Save detailed results to CSV
    results_csv = Path(args.output_folder) / "speed_comparison_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")

def main():
    parser = argparse.ArgumentParser(description='Compare inference speeds between baseline and optimized configurations')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                       help='Input folder with test images')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                       help='Base output folder for all configurations')
    parser.add_argument('-m', '--model_folder', type=str, required=True,
                       help='Model training output folder')
    
    args = parser.parse_args()

    # Create output folder
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    compare_speeds(args)

if __name__ == '__main__':
    main()