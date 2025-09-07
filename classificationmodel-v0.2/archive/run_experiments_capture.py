#!/usr/bin/env python3
"""
Script to run all experiments and capture output to a file
"""

import sys
import os
from datetime import datetime
import subprocess

def run_experiments_with_output_capture():
    """Run all experiments and capture output to a file"""

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiments_output_{timestamp}.txt"

    print(f"🚀 Running all experiments and capturing output to: {output_file}")
    print("=" * 80)

    try:
        # Run the experiments and capture output
        result = subprocess.run(
            [sys.executable, "run_experiments.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Combine stdout and stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\nSTDERR:\n" + result.stderr

        # Add execution info
        header = f"""EXPERIMENTS EXECUTION REPORT
{'='*50}
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Return Code: {result.returncode}
Output File: {output_file}

"""
        full_output = header + full_output

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)

        print(f"✅ Experiments completed!")
        print(f"📁 Output saved to: {output_file}")
        print(f"📊 Return code: {result.returncode}")

        # Print a summary to console
        print("\n" + "="*50)
        print("EXECUTION SUMMARY")
        print("="*50)

        if result.returncode == 0:
            print("✅ All experiments completed successfully")
        else:
            print(f"⚠️  Experiments completed with return code: {result.returncode}")

        # Try to extract final summary from output
        lines = full_output.split('\n')
        summary_found = False
        for i, line in enumerate(lines):
            if "EXPERIMENT SUMMARY" in line:
                summary_found = True
                print("\n📊 EXPERIMENT RESULTS:")
                for j in range(i, min(i+20, len(lines))):
                    if lines[j].strip():
                        print(lines[j])
                break

        if not summary_found:
            print("📝 Full results saved to output file")

        return output_file

    except Exception as e:
        error_msg = f"❌ Error running experiments: {e}"
        print(error_msg)

        # Save error to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENTS EXECUTION REPORT\n{'='*50}\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {error_msg}\n")

        return output_file

if __name__ == "__main__":
    output_file = run_experiments_with_output_capture()
    print(f"\n📄 Complete output available in: {output_file}")
