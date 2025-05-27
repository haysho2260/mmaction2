import os
import subprocess
from pathlib import Path

def get_next_results_number(mmaction_base_dir):
    """Find the next available results file number.
    
    Args:
        mmaction_base_dir (str): Base directory for results
        
    Returns:
        int: Next available number for results file
    """
    i = 0
    while True:
        if not os.path.exists(os.path.join(mmaction_base_dir, f'results_{i}.csv')):
            return i
        i += 1

def run_detection_on_videos(
    root_dir: str,
    filter_tug: bool = True,
    device: str = "cuda:0",
    resume_from: str = None,
    output_csv: str = None
):
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmaction_base_path = Path(script_dir) / "mmaction_result"
    mmaction_base_path.mkdir(exist_ok=True)

    # Use specified output CSV or get next available number
    if output_csv:
        results_file = output_csv
    else:
        results_num = get_next_results_number(mmaction_base_path)
        results_file = os.path.join(mmaction_base_path, f'results_{results_num}.csv')
    
    print(f"\nAll results will be written to: {results_file}")

    root_path = Path(root_dir)
    count = 0
    resume_found = resume_from is None  # If no resume point specified, start from beginning
    
    # Walk recursively
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            # Basic video file check (add more extensions if needed)
            if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue

            if filter_tug and 'tug' not in filename.lower():
                # print(f"Skipping {filename} because it does not contain 'tug'")
                continue
            
            # If we haven't found the resume point yet, check if this is it
            if not resume_found:
                if resume_from in filename:
                    resume_found = True
                else:
                    continue  # Skip until we find the resume point
            
            count += 1
            video_path = Path(dirpath) / filename
            video_path_str = str(video_path.resolve())

            cmd = [
                "python",
                "/code/hchang27/mmaction2/demo/demo_spatiotemporal_det.py",
                video_path_str,
                "--device",
                device,
                "--output-csv",
                results_file
            ]

            print(f"\nProcessing video: {video_path_str}")
            try:
                subprocess.run(cmd, check=True, cwd="/code/hchang27/mmaction2")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_path_str}: {e}")

if __name__ == "__main__":
    # Example usage: To resume from UWisc_1160 and append to results_0.csv
    run_detection_on_videos(
        "/files/pathml/aim2/videos/",
        resume_from="blurred_T28_TUG_2",  # Change this to None to process all files
        output_csv="/code/hchang27/mmaction2/demo/mmaction_result/results_0.csv"
    )
