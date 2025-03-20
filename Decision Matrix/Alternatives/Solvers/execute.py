#!/usr/bin/env python3
import os
import subprocess

def main():
    # Define the shell script lines
    script_lines = []
    
    # --- Environment Initialization ---
    script_lines.append("#!/bin/bash")
    script_lines.append("export LD_LIBRARY_PATH=/mnt/c/Users/javym/Documents/Github/TFGs/Premio1/rc/gurobi910/linux64/lib:$LD_LIBRARY_PATH")
    script_lines.append("export PATH=/mnt/c/Users/javym/Documents/Github/TFGs/Premio1/rc/gurobi910/linux64/bin:$PATH")
    script_lines.append("echo 'Initializing grbgetkey...'")
    script_lines.append("grbgetkey 58e1ed98-92fb-4bf9-afd3-fbc717c34a77")
    script_lines.append("")
    
    # --- Problem Definition ---
    # Change the problem name here (e.g., "X_15" or "C_01")
    script_lines.append("problem=X_15")
    script_lines.append("")
    
    # --- Parameter Lists ---
    # Define the t values and seeds
    script_lines.append("t_list=(20 100 120)")
    script_lines.append("seeds=(42 40 45 10 99)")
    script_lines.append("")
    
    # --- Command Execution ---
    script_lines.append("for t in \"${t_list[@]}\"; do")
    script_lines.append("  for seed in \"${seeds[@]}\"; do")
    # Team1 command (uses problem.json)
    script_lines.append("    echo \"Executing Team1 for t=$t, seed=$seed\"")
    script_lines.append("    ./team1 -t $t -p ${problem}.json -o sol${t}_t1_${problem}_s${seed}.txt -name Team1 -s $seed")
    # Team3 command (uses problem without .json)
    script_lines.append("    echo \"Executing Team3 for t=$t, seed=$seed\"")
    script_lines.append("    ./team3 -t $t -p $problem -o sol${t}_t3_${problem}_s${seed}.txt -name Team3 -s $seed")
    # Team5 command (uses problem.json)
    script_lines.append("    echo \"Executing Team5 for t=$t, seed=$seed\"")
    script_lines.append("    ./team5 -t $t -p ${problem}.json -o sol${t}_t5_${problem}_s${seed}.txt -name Team5 -s $seed")
    script_lines.append("  done")
    script_lines.append("done")
    script_lines.append("")
    
    # --- Organize Output Files ---
    script_lines.append("# Create a base folder (remove underscores from problem name)")
    script_lines.append("base_folder=$(echo ${problem} | tr -d '_')")
    script_lines.append("mkdir -p $base_folder/Team1")
    script_lines.append("mkdir -p $base_folder/Team3")
    script_lines.append("mkdir -p $base_folder/Team5")
    script_lines.append("for file in sol*_t*_$(echo ${problem})_s*.txt; do")
    script_lines.append("  if [[ $file == *\"_t1_\"* ]]; then")
    script_lines.append("    mv $file $base_folder/Team1/")
    script_lines.append("  elif [[ $file == *\"_t3_\"* ]]; then")
    script_lines.append("    mv $file $base_folder/Team3/")
    script_lines.append("  elif [[ $file == *\"_t5_\"* ]]; then")
    script_lines.append("    mv $file $base_folder/Team5/")
    script_lines.append("  fi")
    script_lines.append("done")
    script_lines.append("")
    
    # --- Keep Terminal Open ---
    script_lines.append("echo 'All commands executed. Press Enter to exit.'")
    script_lines.append("read")
    
    # Join lines into a complete script
    script_content = "\n".join(script_lines)
    
    # Save the shell script to a file
    script_filename = "run_all.sh"
    with open(script_filename, "w") as f:
        f.write(script_content)
    
    # Make the shell script executable
    os.chmod(script_filename, 0o755)
    
    # --- Launch a New Terminal Window ---
    # Here we use xterm; modify the command if you prefer another terminal emulator.
    terminal_command = f"xterm -hold -e './{script_filename}'"
    subprocess.Popen(terminal_command, shell=True)

if __name__ == "__main__":
    main()
