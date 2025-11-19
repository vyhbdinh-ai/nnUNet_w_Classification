import time
import subprocess
import pandas as pd
from pathlib import Path
import argparse
import sys
import os

def run_inference(config_name, enable_tta, step_size, input_folder, output_folder, model_folder):    
    # Create output folder for this configuration
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
    # Define configurations: baseline and optimized
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
            "step_size": 0.85,
            "description": "Optimized (TTA disabled, step_size=0.7)"
        }]
    
    input_path = Path(args.input_folder)
    nifti_files = list(input_path.glob("*.nii")) + list(input_path.glob("*.nii.gz"))
    num_cases = len(nifti_files)
    #print(f"Number of cases: {num_cases}")

    results = []
    for config in configurations:
        execution_time, success = run_inference(
            config["name"],
            config["enable_tta"],
            config["step_size"],
            args.input_folder,
            args.output_folder,
            args.model_folder)
        time_per_case = execution_time / num_cases
        results.append({
            "Configuration": config["name"],
            "Description": config["description"],
            "TTA": "Enabled" if config["enable_tta"] else "Disabled",
            "Step Size": config["step_size"],
            "Time (seconds)": round(execution_time, 2),
            "Time per Case (seconds)": round(time_per_case, 2),
            "Status": "Success" if success else "Failed"})
        
    # Calculate speed improvements
    baseline_time = results[0]["Time (seconds)"]

    if len(results) >= 2 and all(r["Status"] == "Success" for r in results):
        baseline = results[0]
        optimized = results[1]
        
        total_speedup = baseline["Time (seconds)"] / optimized["Time (seconds)"]
        per_case_speedup = baseline["Time per Case (seconds)"] / optimized["Time per Case (seconds)"]
        improvement_pct = ((baseline["Time (seconds)"] - optimized["Time (seconds)"]) / baseline["Time (seconds)"]) * 100
        
        results[1]["Improvement %"] = round(improvement_pct, 1)
        results[1]["Speedup Factor"] = round(total_speedup, 2)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Calculate and display summary
    if len(results) >= 2 and all(r["Status"] == "Success" for r in results):
        print(f"\nSUMMARY:")
        print(f"Number of cases: {num_cases}")
        print(f"Baseline (TTA enabled): {baseline['Time per Case (seconds)']:.1f}s per case")
        print(f"Optimized (TTA disabled): {optimized['Time per Case (seconds)']:.1f}s per case")
        print(f"Improvement: {improvement_pct:.1f}% faster")
        print(f"Speedup factor: {total_speedup:.2f}x")
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