#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
import openpyxl
import re

def get_row_priority_action(row):
    """Get the first priority action found in any action label.
    Checks action labels in order (0-4) and returns the first priority action found.
    
    Args:
        row (pd.Series): Single row from DataFrame containing action labels
        
    Returns:
        str or None: First priority action found, or None if no priority action found
    """
    # Check each action label in order (0-4)
    for i in range(5):
        action_col = f'action_label_{i}'
        if action_col in row and pd.notna(row[action_col]):
            action = row[action_col].lower()
            # Return the first priority action found
            if any(key in action for key in ['get up', 'sit', 'bend/bow (at the waist)', 'stand', 'walk']):
                return action
    
    return None

def get_primary_action(frame_rows):
    """Get the primary action for a frame by aggregating all detections in that frame.
    First finds priority action for each detection, then aggregates using priority order.
    Priority order: get up > sit > bend/bow > walk
    
    Args:
        frame_rows (pd.DataFrame): DataFrame containing all rows for a single frame
        
    Returns:
        str or None: Primary action for this frame, or None if no relevant action
    """
    # Get priority action for each row
    row_actions = []
    
    # Get frame number and time information
    frame_num = frame_rows['frames'].iloc[0]
    fps = frame_rows['fps'].iloc[0]
    time_sec = frame_num / fps
    
    for _, row in frame_rows.iterrows():
        action = get_row_priority_action(row)
        if action:
            row_actions.append(action)
    
    print(f"Frame {frame_num} (Time: {time_sec:.2f}s) - Actions: {row_actions}")
    
    # If no priority actions found in any row
    if not row_actions:
        return None
        
    # Aggregate actions by priority
    if 'get up' in row_actions:
        return 'get up'
    elif 'sit' in row_actions:
        return 'sit'
    elif 'bend/bow (at the waist)' in row_actions:
        return 'bend/bow (at the waist)'
    elif 'stand' in row_actions:
        return 'stand'
    elif 'walk' in row_actions:
        return 'walk'
    
    return None

def is_active_state(action):
    """Determine if an action represents an active (standing/walking) state."""
    return action in ['stand', 'walk', 'bend/bow (at the waist)']

def is_inactive_state(action):
    """Determine if an action represents an inactive (sitting/bending) state."""
    return action in ['sit', 'get up']

def find_activity_periods(df, fps):
    """Find the duration between first inactive-to-active transition and last active-to-inactive transition.
    
    Args:
        df (pd.DataFrame): DataFrame containing actions for one video
        fps (float): Frames per second of the video
        
    Returns:
        dict: Dictionary containing the first-to-last activity period, or a period starting at time 0 with sitting state if no transitions found
    """
    # Sort by frame number and group by frame
    df = df.sort_values('frames')
    
    current_state = None
    first_transition_frame = None
    last_transition_frame = None
    
    # Process frame by frame
    for frame_num, frame_group in df.groupby('frames'):
        current_action = get_primary_action(frame_group)
        if not current_action:
            continue
            
        # Initialize state if None
        if current_state is None:
            current_state = current_action
            continue
            
        # If we transition from inactive to active state
        if is_inactive_state(current_state) and is_active_state(current_action):
            # Record first transition if we haven't seen one yet
            if first_transition_frame is None:
                first_transition_frame = frame_num
            
        # If we transition from active to inactive state
        elif is_active_state(current_state) and is_inactive_state(current_action):
            # Always update last transition when we see one
            last_transition_frame = frame_num
            
        current_state = current_action
    
    # If we found both transitions, return the period
    if first_transition_frame is not None and last_transition_frame is not None:
        duration = (last_transition_frame - first_transition_frame) / fps
        return [{
            'start_frame': first_transition_frame,
            'end_frame': last_transition_frame,
            'duration': duration,
            'start_time': first_transition_frame / fps,
            'end_time': last_transition_frame / fps
        }]
        
    # Handle case where video doesn't register sitting to begin
    # elif first_transition_frame is None and last_transition_frame is not None:
    #     duration = (last_transition_frame - 0) / fps
    #     return [{
    #         'start_frame': 0,
    #         'end_frame': last_transition_frame,
    #         'duration': duration,
    #         'start_time': 0,
    #         'end_time': last_transition_frame / fps
    #     }]
    
    # Handle case where video ends during active period
    elif first_transition_frame is not None and is_active_state(current_state):
        last_frame = df['frames'].max()
        duration = (last_frame - first_transition_frame) / fps
        return [{
            'start_frame': first_transition_frame,
            'end_frame': last_frame,
            'duration': duration,
            'start_time': first_transition_frame / fps,
            'end_time': last_frame / fps
        }]
    
    # If no activity period found, assume sitting at time 0
    return []

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
        
    # Extract test number if present (e.g., tug_3, tug2, etc.)
    test_number = None
    test_number_match = re.search(r'[_-]?(\d+)(?=\.|$)', filename.split('.')[0])
    if test_number_match:
        test_number = test_number_match.group(1)
    
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
            base_patterns = patterns
            # If we have a test number, add more specific patterns
            if test_number:
                # Add patterns with different number formats
                number_patterns = [
                    f'*{p[:-1]}{test_number}*' for p in patterns  # e.g., *tug3*
                ] + [
                    f'*{p[:-1]}_{test_number}*' for p in patterns  # e.g., *tug_3*
                ] + [
                    f'*{p[:-1]}-{test_number}*' for p in patterns  # e.g., *tug-3*
                ]
                # Put more specific patterns first
                search_patterns.extend(number_patterns)
            else:
                print(f"no test number in filename {filename}")
                # If no test number in filename, try both unnumbered and number 1 patterns
                number_one_patterns = [
                    f'*{p[:-1]}1*' for p in patterns  # e.g., *tug1*
                ] + [
                    f'*{p[:-1]}_1*' for p in patterns  # e.g., *tug_1*
                ] + [
                    f'*{p[:-1]}-1*' for p in patterns  # e.g., *tug-1*
                ]
                # Put base patterns first for unnumbered files, then try number 1 patterns
                search_patterns.extend(base_patterns)
                search_patterns.extend(number_one_patterns)
            
            # Add base patterns as fallback if we had a test number
            if test_number:
                search_patterns.extend(base_patterns)
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
                # If we found a match and we're using base patterns, print a warning if it contains a number
                if test_number is None and any(re.search(r'(?:^|[_-])1(?:\D|$)', f) for f in matching_files):
                    print(f"Warning: Unnumbered video file {filename} matched with a numbered ground truth file {matching_files[0]}")
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
        
        # Find first occurrence of start marker
        for marker in start_markers:
            mask = df[behavior_col].str.contains(marker, na=False, case=False)
            if mask.any():
                start_row = df[mask].iloc[0]
                break
                
        # Find last occurrence of stop marker
        for marker in stop_markers:
            mask = df[behavior_col].str.contains(marker, na=False, case=False)
            if mask.any():
                stop_row = df[mask].iloc[-1]  # Changed from iloc[0] to iloc[-1]
                break
        
        if start_row is None or stop_row is None:
            print(f"Could not find test start/stop markers in {excel_path}")
            return None
        
        # Try different time column names, prioritizing the most precise ones
        time_cols = ['Time_Relative_sf', 'Time_Relative_s', 'time_relative_s', 'Time', 'time']
        frame_cols = ['Time_Relative_f', 'time_relative_f', 'frames', 'frames']
        
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

def interpolate_missing_tracks(df):
    """Fill in missing frames for each track ID by interpolating between known frames.
    
    Args:
        df (pd.DataFrame): DataFrame containing action detections
        
    Returns:
        pd.DataFrame: DataFrame with interpolated rows for missing frames
    """
    # Sort by track_id and frame
    df = df.sort_values(['track_id', 'frames'])
    
    # Create a list to store interpolated rows
    interpolated_rows = []
    
    # Process each track_id separately
    for track_id, track_df in df.groupby('track_id'):
        frames = track_df['frames'].values
        
        # Find gaps in frames
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            frame_gap = next_frame - current_frame
            
            # If there's a small gap (e.g., 5 frames or less), interpolate
            if 1 < frame_gap <= 5:
                print(f"\nInterpolating track_id {track_id} between frames {current_frame} and {next_frame}")
                
                # Get the rows before and after the gap
                before_row = track_df[track_df['frames'] == current_frame].iloc[0]
                after_row = track_df[track_df['frames'] == next_frame].iloc[0]
                
                # Get priority actions for before and after frames
                before_action = get_row_priority_action(before_row)
                after_action = get_row_priority_action(after_row)
                print(f"  Before gap: {before_action}, After gap: {after_action}")
                
                # Only interpolate if the actions are consistent or follow logical progression
                should_interpolate = True
                if before_action != after_action:
                    # If walking on either side, don't introduce sitting/bending
                    if ('walk' in [before_action, after_action] and 
                        any(a in ['sit', 'bend/bow (at the waist)'] for a in [before_action, after_action])):
                        print(f"  Skipping interpolation due to inconsistent actions: {before_action} -> {after_action}")
                        should_interpolate = False
                
                if should_interpolate:
                    # Interpolate for each missing frame
                    for missing_frame in range(int(current_frame + 1), int(next_frame)):
                        # Calculate interpolation factor
                        alpha = (missing_frame - current_frame) / frame_gap
                        
                        # Create interpolated row
                        new_row = before_row.copy()
                        new_row['frames'] = missing_frame
                        
                        # Interpolate bounding box coordinates
                        for coord in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']:
                            new_row[coord] = int(before_row[coord] * (1 - alpha) + after_row[coord] * alpha)
                        
                        # For actions, use the higher priority action if they're different
                        if before_action == after_action:
                            # If actions are the same, keep them
                            action_to_use = before_action
                        else:
                            # If actions are different, use the one with higher priority
                            priority_order = ['get up', 'sit', 'bend/bow (at the waist)', 'walk']
                            before_priority = priority_order.index(before_action) if before_action in priority_order else len(priority_order)
                            after_priority = priority_order.index(after_action) if after_action in priority_order else len(priority_order)
                            action_to_use = before_action if before_priority <= after_priority else after_action
                        
                        # Set the interpolated action
                        new_row['action_label_0'] = action_to_use
                        # Clear other action labels to avoid conflicts
                        for i in range(1, 5):
                            new_row[f'action_label_{i}'] = None
                            new_row[f'action_score_{i}'] = 0.0
                        
                        interpolated_rows.append(new_row)
                        print(f"  Added interpolated frame {missing_frame} with action {action_to_use}")
    
    # If we have interpolated rows, add them to the original DataFrame
    if interpolated_rows:
        interpolated_df = pd.DataFrame(interpolated_rows)
        df = pd.concat([df, interpolated_df], ignore_index=True)
        df = df.sort_values(['track_id', 'frames'])
    
    return df

def analyze_video_actions(df, filename, results_list):
    """Analyze action transitions for a specific video.
    
    Args:
        df (pd.DataFrame): DataFrame containing actions for one video
        filename (str): Name of the video being analyzed
        results_list (list): List to store results for CSV output
    """
    fps = df['fps'].iloc[0]  # Get fps from first row
    
    print(f"\nAnalyzing video: {filename}")
    
    # Interpolate missing frames before analysis
    print("\nChecking for missing frames and interpolating...")
    df = interpolate_missing_tracks(df)
    
    # Find activity periods from our detection
    activity_periods = find_activity_periods(df, fps)
    
    # Initialize result dictionary
    result = {
        'filename': filename,
        'ground_truth_found': False,
        'ground_truth_file': None,  # Initialize ground truth file path
        'ground_truth_label_file': None  # Initialize ground truth label file path
    }
    
    if activity_periods:
        print("\nDetected Activity Periods (Standing/Walking between Sitting/Bending):")
        # Use the first period for comparison (assuming single activity period per video)
        period = activity_periods[0]
        result.update({
            'detected_start_time': round(period['start_time'], 2),
            'detected_end_time': round(period['end_time'], 2),
            'detected_duration': round(period['duration'], 2)
        })
        
        print(f"\nPredicted:")
        print(f"  Start Time: {period['start_time']:.2f} seconds")
        print(f"  End Time: {period['end_time']:.2f} seconds")
        print(f"  Duration: {period['duration']:.2f} seconds")
    else:
        print("\nNo activity periods detected.")
        result.update({
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
            result['ground_truth_file'] = df['ground_truth_file'].iloc[0] if 'ground_truth_file' in df.columns else None
            result['ground_truth_label_file'] = ground_truth_file
            result.update({
                'ground_truth_start_time': round(ground_truth['start_time'], 2),
                'ground_truth_end_time': round(ground_truth['stop_time'], 2),
                'ground_truth_duration': round(ground_truth['duration'], 2)
            })
            
            # Calculate differences if we have both detection and ground truth
            if activity_periods:
                result.update({
                    'start_time_diff': round(period['start_time'] - ground_truth['start_time'], 2),
                    'end_time_diff': round(period['end_time'] - ground_truth['stop_time'], 2),
                    'duration_diff': round(period['duration'] - ground_truth['duration'], 2)
                })
            
            print(f"  Test Start Time: {ground_truth['start_time']:.2f} seconds")
            print(f"  Test Stop Time: {ground_truth['stop_time']:.2f} seconds")
            print(f"  Test Duration: {ground_truth['duration']:.2f} seconds")
    else:
        print("\nNo ground truth file found.")
        result.update({
            'ground_truth_start_time': None,
            'ground_truth_end_time': None,
            'ground_truth_duration': None,
            'start_time_diff': None,
            'end_time_diff': None,
            'duration_diff': None
        })
    
    results_list.append(result)

def main():
    parser = argparse.ArgumentParser(description='Analyze action transitions from CSV results.')
    parser.add_argument('--csv_path', default='/code/hchang27/mmaction2/demo/mmaction_result/video_with_large_errs.csv', help='Path to the CSV file containing action detection results')
    # parser.add_argument('--csv_path', default='/code/hchang27/mmaction2/demo/mmaction_result/results_0.csv', help='Path to the CSV file containing action detection results')
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