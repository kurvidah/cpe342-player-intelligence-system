
import os
import pandas as pd

def compile_submission(sample_submission_path, tasks_dir, output_path):
    if not os.path.exists(sample_submission_path):
        print(f"Error: Sample submission file not found at '{sample_submission_path}'")
        return

    # Read the sample submission to get the base structure and 'id' column
    try:
        submission_df = pd.read_csv(sample_submission_path)
    except Exception as e:
        print(f"Error reading sample submission file: {e}")
        return

    # List all task directories (e.g., 'task1', 'task2', ...)
    try:
        task_folders = sorted([d for d in os.listdir(tasks_dir) if os.path.isdir(os.path.join(tasks_dir, d)) and d.startswith('task')])
    except FileNotFoundError:
        print(f"Error: Tasks directory not found at '{tasks_dir}'")
        return

    # Iterate through each task folder to read its submission file
    for task_folder in task_folders:
        task_name = task_folder.replace('task', 'task') # e.g. 'task1'
        submission_file_path = os.path.join(tasks_dir, task_folder, 'submission.csv')

        if os.path.exists(submission_file_path):
            try:
                # Read the prediction file for the current task
                task_submission_df = pd.read_csv(submission_file_path)

                # Ensure the prediction file has 'id' and 'prediction' columns
                if 'id' not in task_submission_df.columns or 'prediction' not in task_submission_df.columns:
                    print(f"Warning: '{submission_file_path}' is missing 'id' or 'prediction' column. Skipping.")
                    continue
                
                # Rename 'prediction' column to the corresponding task name (e.g., 'task1')
                task_submission_df = task_submission_df.rename(columns={'prediction': task_name})

                # Merge the task predictions into the main submission DataFrame
                if task_name in submission_df.columns:
                    submission_df = submission_df.drop(columns=[task_name])
                submission_df = pd.merge(submission_df, task_submission_df[['id', task_name]], on='id', how='left')

            except Exception as e:
                print(f"Error processing '{submission_file_path}': {e}")
        else:
            print(f"Warning: Submission file for {task_name} not found at '{submission_file_path}'. The column will be empty.")

    # Save the final compiled submission file
    try:
        submission_df.to_csv(output_path, index=False)
        print(f"Successfully generated submission file at '{output_path}'")
    except Exception as e:
        print(f"Error writing final submission file: {e}")

if __name__ == '__main__':
    # Define file and directory paths
    SAMPLE_SUBMISSION_PATH = 'sample_submission.csv'
    TASKS_DIR = '.' # Assumes task folders are in the same directory as the script
    OUTPUT_PATH = 'submission.csv'

    # Run the compilation process
    compile_submission(SAMPLE_SUBMISSION_PATH, TASKS_DIR, OUTPUT_PATH)
