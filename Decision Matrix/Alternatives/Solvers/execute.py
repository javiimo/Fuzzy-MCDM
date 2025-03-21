# WARNING:
# This code uses windows subsystem linux to run the solvers since they were built for debian 10. It shouldn't be difficult to adapt it to run just for regular linux. Paths would need to be adapted to run on a new computer. 
# The problem with wsl is that the hostid changes from time to time, so that is why we would need to frequently create new keys.

import os
import subprocess

def create_run_all_sh():
    # Build the Linux shell script that will be executed under WSL.
    lines = [
        "#!/bin/bash",
        "# Redirect stdout and stderr to execution.log (while still displaying on terminal)",
        "exec > >(tee execution.log) 2>&1",
        "",
        "# --- Environment Initialization ---",
        "export LD_LIBRARY_PATH=/mnt/c/Users/javym/Documents/Github/TFGs/Premio1/rc/gurobi910/linux64/lib:$LD_LIBRARY_PATH",
        "export PATH=/mnt/c/Users/javym/Documents/Github/TFGs/Premio1/rc/gurobi910/linux64/bin:$PATH",
        "echo 'Initializing grbgetkey...'",
        "gurobi_cl --version",
        #"gurobi_cl test.lp  ", # If it gives a file error, then the license is working. Otherwise, a license needs to be set up
        "grbgetkey <key>", # Set a key 
        "",
        "# --- Problem Definition (change here to update the problem name) ---",
        "problem=X_12",
        "",
        "# --- Parameter Lists ---",
        "t_list=(180 300 500 700 900)",
        "seeds=(42 33 73)",
        "",
        "# --- Execute Team Commands ---",
        "for t in \"${t_list[@]}\"; do",
        "  for seed in \"${seeds[@]}\"; do",
        "    echo \"=======================================================================\"",
        "    echo \"EXECUTING TEAM1 FOR T=$t, SEED=$seed\"",
        "    echo \"=======================================================================\"",
        "    ./team1 -t $t -p ${problem}.json -o sol${t}_t1_${problem}_s${seed}.txt -name Team1 -s $seed",
        "    echo \"=======================================================================\"",
        "    echo \"EXECUTING TEAM3 FOR T=$t, SEED=$seed\"",
        "    echo \"=======================================================================\"",
        "    ./team3 -t $t -p $problem -o sol${t}_t3_${problem}_s${seed}.txt -name Team3 -s $seed",
        "    echo \"=======================================================================\"",
        "    echo \"EXECUTING TEAM5 FOR T=$t, SEED=$seed\"",
        "    echo \"=======================================================================\"",
        "    ./team5 -t $t -p ${problem}.json -o sol${t}_t5_${problem}_s${seed}.txt -s $seed",
        "  done",
        "done",
        "",
        "# --- Organize Output Files ---",
        "base_folder=$(echo ${problem} | tr -d '_')",
        "mkdir -p $base_folder/Team1",
        "mkdir -p $base_folder/Team3",
        "mkdir -p $base_folder/Team5",
        "for file in sol*_t*_$(echo ${problem})_s*.txt; do",
        "  if [[ $file == *\"_t1_\"* ]]; then",
        "    mv $file $base_folder/Team1/",
        "  elif [[ $file == *\"_t3_\"* ]]; then",
        "    mv $file $base_folder/Team3/",
        "  elif [[ $file == *\"_t5_\"* ]]; then",
        "    mv $file $base_folder/Team5/",
        "  fi",
        "done",
        "",
        "# --- Keep Terminal Open for Inspection ---",
        "echo 'All commands executed. Press Enter to exit.'",
        "read"
    ]
    script_content = "\n".join(lines)
    script_filename = "run_all.sh"
    # Write with Unix line endings
    with open(script_filename, "w", newline="\n") as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    return script_filename

def create_run_all_bat():
    # Batch file to convert the current Windows directory to a WSL path and run the Linux shell script.
    bat_lines = [
        "@echo off",
        "REM Convert current directory to WSL path",
        "for /f \"usebackq tokens=*\" %%i in (`wsl wslpath \"%CD%\"`) do set WSL_PATH=%%i",
        "echo WSL Path is %WSL_PATH%",
        "REM Run the shell script via WSL",
        "wsl bash \"%WSL_PATH%/run_all.sh\"",
        "pause"
    ]
    bat_content = "\r\n".join(bat_lines)
    bat_filename = "run_all.bat"
    with open(bat_filename, "w") as f:
        f.write(bat_content)
    return bat_filename

def main():
    # Create the Linux shell script.
    sh_script = create_run_all_sh()
    print(f"Created Linux shell script: {sh_script}")
    
    # Create the Windows batch file.
    bat_script = create_run_all_bat()
    print(f"Created Windows batch file: {bat_script}")
    
    # Launch a new Command Prompt window that runs the batch file.
    cmd_command = f'start cmd /k "{bat_script}"'
    print("Launching a new CMD window with WSL execution...")
    subprocess.Popen(cmd_command, shell=True)

if __name__ == "__main__":
    main()
