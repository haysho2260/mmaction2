#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
import openpyxl

def get_row_priority_action(row):
    """Get the priority action for a single row, checking all action labels.
    Priority order: getup > sit > bend/bow > walk
    
    Args:
        row (pd.Series): Single row from DataFrame containing action labels
        
    Returns:
        str or None: Priority action for this row, or None if no priority action found
    """
    # First check action_label_0
    if 'action_label_0' in row and pd.notna(row['action_label_0']):
        action = row['action_label_0'].lower()
        if 'getup' in action:
            return 'getup'
        elif 'sit' in action:
            return 'sit'
        elif 'bend/bow (at the waist)' in action:
            return 'bend/bow (at the waist)'
        elif 'walk' in action:
            return 'walk'
    
    # If no priority action in action_label_0, check other labels
    for i in range(1, 5):
        action_col = f'action_label_{i}'
        if action_col in row and pd.notna(row[action_col]):
            action = row[action_col].lower()
            if 'getup' in action:
                return 'getup'
            elif 'sit' in action:
                return 'sit'
            elif 'bend/bow (at the waist)' in action:
                return 'bend/bow (at the waist)'
            elif 'walk' in action:
                return 'walk'
    
    return None

def get_primary_action(frame_rows):
    """Get the primary action for a frame by aggregating all detections in that frame.
    First finds priority action for each detection, then aggregates using priority order.
    Priority order: getup > sit > bend/bow > walk
    
    Args:
        frame_rows (pd.DataFrame): DataFrame containing all rows for a single frame
        
    Returns:
        str or None: Primary action for this frame, or None if no relevant action
    """
    # Get priority action for each row
    row_actions = []
    for _, row in frame_rows.iterrows():
        action = get_row_priority_action(row)
        if action:
            row_actions.append(action)
    
    # If no priority actions found in any row
    if not row_actions:
        return None
        
    # Aggregate actions by priority
    if 'getup' in row_actions:
        return 'getup'
    elif 'sit' in row_actions:
        return 'sit'
    elif 'bend/bow (at the waist)' in row_actions:
        return 'bend/bow (at the waist)'
    elif 'walk' in row_actions:
        return 'walk'
    
    return None

def is_active_state(action):
    """Determine if an action represents an active (standing/walking) state."""
    return action in ['stand', 'walk']

def is_inactive_state(action):
    """Determine if an action represents an inactive (sitting/bending) state."""
    return action in ['sit', 'getup', 'bend/bow (at the waist)']

def find_activity_periods(df, fps):
    """Find periods of activity (standing/walking) between inactive states.
    
    Args:
        df (pd.DataFrame): DataFrame containing actions for one video
        fps (float): Frames per second of the video
        
    Returns:
        list: List of dictionaries containing activity periods
    """
    # Sort by frame number and group by frame
    df = df.sort_values('frame')
    activity_periods = []
    
    current_state = None
    activity_start_frame = None
    
    # Process frame by frame
    for frame_num, frame_group in df.groupby('frame'):
        current_action = get_primary_action(frame_group)
        if not current_action:
            continue
            
        # Detect transitions between active and inactive states
        if current_state is None:
            current_state = current_action
            continue
            
        # If we transition from inactive to active state
        if is_inactive_state(current_state) and is_active_state(current_action):
            activity_start_frame = frame_num
            
        # If we transition from active to inactive state and we had a start frame
        elif is_active_state(current_state) and is_inactive_state(current_action) and activity_start_frame is not None:
            duration = (frame_num - activity_start_frame) / fps
            activity_periods.append({
                'start_frame': activity_start_frame,
                'end_frame': frame_num,
                'duration': duration,
                'start_time': activity_start_frame / fps,
                'end_time': frame_num / fps
            })
            activity_start_frame = None
            
        current_state = current_action
    
    # Handle case where video ends during active period
    if activity_start_frame is not None and is_active_state(current_state):
        last_frame = df['frame'].max()
        duration = (last_frame - activity_start_frame) / fps
        activity_periods.append({
            'start_frame': activity_start_frame,
            'end_frame': last_frame,
            'duration': duration,
            'start_time': activity_start_frame / fps,
            'end_time': last_frame / fps
        })
    
    return activity_periods

def extract_video_info(filename):
    """Extract institution and participant ID from various video filename formats.
    
    Args:
        filename (str): Name of the video file (e.g., 'UWisc_1068_TUG.mp4', 'P13_TUG.mp4', 'blurred_T07_STS.MOV')
        
    Returns:
        tuple: (institution, participant_id, test_type) or (None, None, None) if not recognized
    """
    # Remove file extension and any "blurred_" prefix
    base_name = os.path.splitext(filename)[0]
    base_name = base_name.replace('blurred_', '')
    
    # Try different formats
    if base_name.startswith('UWisc_'):
        # Format: UWisc_1068_TUG
        parts = base_name.split('_')
        return ('UWisc', parts[1], parts[2].lower())
    elif base_name.startswith('P'):
        # Format: P13_TUG
        parts = base_name.split('_')
        participant_num = parts[0][1:]  # Remove 'P' prefix
        return ('CP', participant_num, parts[1].lower())
    elif base_name.startswith('T'):
        # Format: T07_STS
        parts = base_name.split('_')
        return ('VA', parts[0][1:], parts[1].lower())  # Remove 'T' prefix
    
    return None, None, None

def find_ground_truth_file(filename):
    """Find the corresponding ground truth file for a video.
    
    Args:
        filename (str): Name of the video file
        
    Returns:
        str or None: Path to ground truth file if found, None otherwise
    """
    institution, participant_id, test_type = extract_video_info(filename)
    if not institution:
        print(f"Could not parse video name format: {filename}")
        return None
        
    # Map test types to search patterns
    test_patterns = {
        'tug': ['*tug*', '*TUG*'],
        'sts': ['*sts*', '*STS*', '*s2s*', '*S2S*'],
        'walk': ['*walk*', '*WALK*', '*10m*'],
        'balance': ['*balance*', '*BALANCE*', '*balc*', '*BALC*']
    }
    
    # Get the appropriate search patterns
    search_patterns = []
    for test_key, patterns in test_patterns.items():
        if test_key in test_type.lower():
            search_patterns.extend(patterns)
            break
    
    if not search_patterns:
        print(f"Unknown test type: {test_type}")
        return None
    
    # Construct path to ground truth directory
    ground_truth_dir = f'/files/pathml/aim2/raw_labels/{institution}'
    
    # Try different participant ID formats
    id_patterns = []
    if institution == 'UWisc':
        id_patterns.append(participant_id)
    elif institution == 'CP':
        # Try both P13 and 13 formats
        id_patterns.extend([f'P{participant_id}', participant_id])
    elif institution == 'VA':
        # Try both T07 and 07 formats
        id_patterns.extend([f'T{participant_id}', participant_id])
    
    # Try all combinations of patterns
    for id_pat in id_patterns:
        for test_pat in search_patterns:
            pattern = f"{ground_truth_dir}/*{id_pat}*{test_pat}*.xlsx"
            matching_files = glob.glob(pattern)
            if matching_files:
                return matching_files[0]
    
    print(f"No matching ground truth file found for {institution} participant {participant_id} {test_type} test")
    return None

def get_ground_truth_times(excel_path):
    """Extract test start and stop times from ground truth Excel file.
    
    Args:
        excel_path (str): Path to the ground truth Excel file
        
    Returns:
        dict: Dictionary containing timing information or None if not found
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Try different column names for behavior
        behavior_cols = ['Behavior', 'behavior', 'BEHAVIOR']
        behavior_col = next((col for col in behavior_cols if col in df.columns), None)
        if not behavior_col:
            print(f"Could not find behavior column in {excel_path}")
            return None
            
        # Try different test marker formats
        start_markers = ['Test_start', 'TEST_START', 'test_start', 'Start', 'START']
        stop_markers = ['Test_stop', 'TEST_STOP', 'test_stop', 'Stop', 'STOP']
        
        start_row = None
        stop_row = None
        
        for marker in start_markers:
            mask = df[behavior_col].str.contains(marker, na=False, case=False)
            if mask.any():
                start_row = df[mask].iloc[0]
                break
                
        for marker in stop_markers:
            mask = df[behavior_col].str.contains(marker, na=False, case=False)
            if mask.any():
                stop_row = df[mask].iloc[0]
                break
        
        if start_row is None or stop_row is None:
            print(f"Could not find test start/stop markers in {excel_path}")
            return None
        
        # Try different time column names, prioritizing the most precise ones
        time_cols = ['Time_Relative_sf', 'Time_Relative_s', 'time_relative_s', 'Time', 'time']
        frame_cols = ['Time_Relative_f', 'time_relative_f', 'Frame', 'frame']
        
        time_col = next((col for col in time_cols if col in df.columns), None)
        frame_col = next((col for col in frame_cols if col in df.columns), None)
        
        if not time_col:
            print(f"Could not find time column in {excel_path}")
            return None
        
        # Get timing information
        start_time = float(start_row[time_col])  # Convert to float to handle any string formats
        stop_time = float(stop_row[time_col])
        
        result = {
            'start_time': start_time,
            'stop_time': stop_time,
            'duration': stop_time - start_time
        }
        
        # Add frame information if available
        if frame_col:
            result.update({
                'start_frame': int(start_row[frame_col]),  # Convert to int for frame numbers
                'stop_frame': int(stop_row[frame_col])
            })
            
        return result
        
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return None

def analyze_video_actions(df, filename, results_list):
    """Analyze action transitions for a specific video.
    
    Args:
        df (pd.DataFrame): DataFrame containing actions for one video
        filename (str): Name of the video being analyzed
        results_list (list): List to store results for CSV output
    """
    fps = df['fps'].iloc[0]  # Get fps from first row
    
    print(f"\nAnalyzing video: {filename}")
    
    # Find activity periods from our detection
    activity_periods = find_activity_periods(df, fps)
    
    # Initialize result dictionary
    result = {
        'filename': filename,
        'detected_periods': len(activity_periods),
        'ground_truth_found': False
    }
    
    if activity_periods:
        print("\nDetected Activity Periods (Standing/Walking between Sitting/Bending):")
        # Use the first period for comparison (assuming single activity period per video)
        period = activity_periods[0]
        result.update({
            'detected_start_frame': period['start_frame'],
            'detected_end_frame': period['end_frame'],
            'detected_start_time': round(period['start_time'], 2),
            'detected_end_time': round(period['end_time'], 2),
            'detected_duration': round(period['duration'], 2)
        })
        
        print(f"\nPeriod:")
        print(f"  Start Frame: {period['start_frame']}")
        print(f"  End Frame: {period['end_frame']}")
        print(f"  Start Time: {period['start_time']:.2f} seconds")
        print(f"  End Time: {period['end_time']:.2f} seconds")
        print(f"  Duration: {period['duration']:.2f} seconds")
    else:
        print("\nNo activity periods detected.")
        result.update({
            'detected_start_frame': None,
            'detected_end_frame': None,
            'detected_start_time': None,
            'detected_end_time': None,
            'detected_duration': None
        })
    
    # Get ground truth data
    ground_truth_file = find_ground_truth_file(filename)
    if ground_truth_file:
        print("\nGround Truth Data:")
        ground_truth = get_ground_truth_times(ground_truth_file)
        if ground_truth:
            result['ground_truth_found'] = True
            result.update({
                'ground_truth_start_frame': ground_truth.get('start_frame'),
                'ground_truth_end_frame': ground_truth.get('stop_frame'),
                'ground_truth_start_time': round(ground_truth['start_time'], 2),
                'ground_truth_end_time': round(ground_truth['stop_time'], 2),
                'ground_truth_duration': round(ground_truth['duration'], 2)
            })
            
            # Calculate differences if we have both detection and ground truth
            if activity_periods:
                result.update({
                    'start_frame_diff': period['start_frame'] - ground_truth['start_frame'] if 'start_frame' in ground_truth else None,
                    'end_frame_diff': period['end_frame'] - ground_truth['stop_frame'] if 'stop_frame' in ground_truth else None,
                    'start_time_diff': round(period['start_time'] - ground_truth['start_time'], 2),
                    'end_time_diff': round(period['end_time'] - ground_truth['stop_time'], 2),
                    'duration_diff': round(period['duration'] - ground_truth['duration'], 2)
                })
            
            if ground_truth.get('start_frame') is not None:
                print(f"  Start Frame: {ground_truth['start_frame']}")
                print(f"  Stop Frame: {ground_truth['stop_frame']}")
            print(f"  Test Start Time: {ground_truth['start_time']:.2f} seconds")
            print(f"  Test Stop Time: {ground_truth['stop_time']:.2f} seconds")
            print(f"  Test Duration: {ground_truth['duration']:.2f} seconds")
    else:
        print("\nNo ground truth file found.")
        result.update({
            'ground_truth_start_frame': None,
            'ground_truth_end_frame': None,
            'ground_truth_start_time': None,
            'ground_truth_end_time': None,
            'ground_truth_duration': None,
            'start_frame_diff': None,
            'end_frame_diff': None,
            'start_time_diff': None,
            'end_time_diff': None,
            'duration_diff': None
        })
    
    results_list.append(result)

def main():
    parser = argparse.ArgumentParser(description='Analyze action transitions from CSV results.')
    parser.add_argument('--csv_path', default='/code/hchang27/mmaction2/demo/mmaction_result/results_0.csv', help='Path to the CSV file containing action detection results')
    args = parser.parse_args()
    
    # Read the CSV file
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {args.csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create list to store results
    results_list = []
    
    # Group by filename and analyze each video separately
    for filename, video_df in df.groupby('filename'):
        analyze_video_actions(video_df, filename, results_list)
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    
    # Generate output filename based on input filename
    input_basename = os.path.basename(args.csv_path)
    input_name = os.path.splitext(input_basename)[0]
    output_path = os.path.join(os.path.dirname(args.csv_path), f'{input_name}_analysis.csv')
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nAnalysis results saved to: {output_path}")

if __name__ == '__main__':
    main() 