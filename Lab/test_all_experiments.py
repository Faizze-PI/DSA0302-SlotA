"""
Master Test Harness: Automated Verification of All 26 NLP Lab Experiments
Course: DSA0302 - Natural Language Processing
University: SIMATS Engineering
"""

import os
import sys
import subprocess
import time

def run_tests():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    programs_dir = os.path.join(base_dir, "Programs")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    experiments = sorted([d for d in os.listdir(programs_dir) if os.path.isdir(os.path.join(programs_dir, d))])
    
    print("=" * 85)
    print("      DSA0302: NATURAL LANGUAGE PROCESSING - MASTER TEST HARNESS      ")
    print("=" * 85)
    print(f"Total Experiments Discovered: {len(experiments)}")
    print(f"Target Programs Directory   : {programs_dir}")
    print(f"Output Transcripts Directory: {outputs_dir}")
    print("-" * 85)
    print(f"{'#':<4} | {'Experiment Name':<32} | {'Time (s)':<10} | {'Status':<10} | {'Exit Code'}")
    print("-" * 85)

    passed_count = 0
    failed_count = 0
    results = []

    for exp_folder in experiments:
        exp_path = os.path.join(programs_dir, exp_folder)
        main_script = os.path.join(exp_path, "main.py")
        exp_num = exp_folder[:2]

        if not os.path.exists(main_script):
            print(f"{exp_num:<4} | {exp_folder:<32} | {'0.00':<10} | {'FAILED':<10} | main.py missing")
            failed_count += 1
            continue

        start_time = time.time()
        try:
            # Execute script inside its directory
            proc = subprocess.run(
                [sys.executable, "main.py"],
                cwd=exp_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            elapsed = time.time() - start_time
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode

            combined_output = stdout
            if stderr:
                combined_output += f"\n[STDERR]:\n{stderr}"

            # Save transcript to Lab/outputs/ExpXX_Output.txt
            out_file = os.path.join(outputs_dir, f"Exp{exp_num}_Output.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(combined_output)

            if returncode == 0:
                status = "PASS"
                passed_count += 1
            else:
                status = "FAIL"
                failed_count += 1

            print(f"{exp_num:<4} | {exp_folder:<32} | {elapsed:<10.2f} | {status:<10} | {returncode}")
            results.append((exp_num, exp_folder, elapsed, status, returncode))

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"{exp_num:<4} | {exp_folder:<32} | {elapsed:<10.2f} | {'TIMEOUT':<10} | -1")
            failed_count += 1
            results.append((exp_num, exp_folder, elapsed, "TIMEOUT", -1))
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{exp_num:<4} | {exp_folder:<32} | {elapsed:<10.2f} | {'ERROR':<10} | {str(e)}")
            failed_count += 1
            results.append((exp_num, exp_folder, elapsed, "ERROR", str(e)))

    print("-" * 85)
    print(f"VERIFICATION SUMMARY: Total: {len(experiments)} | PASSED: {passed_count} | FAILED: {failed_count}")
    pass_rate = (passed_count / len(experiments)) * 100 if experiments else 0
    print(f"OVERALL SUCCESS RATE: {pass_rate:.1f}%")
    print("=" * 85)

    return failed_count == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
