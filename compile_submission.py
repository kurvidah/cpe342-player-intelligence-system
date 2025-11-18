import os
import pandas as pd

def compile_submission(submission_path, tasks_dir):
    if not os.path.exists(submission_path):
        print(f"Error: Submission file not found at '{submission_path}'")
        return

    # Read the existing submission to get the base structure and original values
    try:
        submission_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"Error reading submission file: {e}")
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

                # Identify the prediction column (the one that is not 'id')
                prediction_columns = [col for col in task_submission_df.columns if col != 'id']
                if not prediction_columns:
                    print(f"Warning: '{submission_file_path}' is missing a prediction column. Skipping this task.")
                    continue
                
                prediction_column = prediction_columns[0]

                # Rename 'prediction' column to the corresponding task name (e.g., 'task1')
                task_submission_df = task_submission_df.rename(columns={prediction_column: task_name})

                # Update the main submission DataFrame with the new predictions
                if task_name in submission_df.columns:
                    # Create a temporary dataframe with the new predictions
                    temp_df = submission_df[['id']].merge(task_submission_df[['id', task_name]], on='id', how='left')
                    # Update the original dataframe
                    submission_df[task_name] = temp_df[task_name]
                else:
                    print(f"Warning: Column '{task_name}' not found in '{submission_path}'. Skipping update for this task.")

            except Exception as e:
                print(f"Error processing '{submission_file_path}': {e}")
        else:
            print(f"Info: Submission file for {task_name} not found at '{submission_file_path}'. Keeping original values.")

    # Save the final compiled submission file
    try:
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully updated submission file at '{submission_path}'")
    except Exception as e:
        print(f"Error writing final submission file: {e}")

if __name__ == '__main__':
    # Define file and directory paths
    SUBMISSION_PATH = 'submission.csv'
    TASKS_DIR = '.' # Assumes task folders are in the same directory as the script

    # Run the compilation process
    compile_submission(SUBMISSION_PATH, TASKS_DIR)
