import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    """
    Performs exploratory data analysis on the dataframe.
    """
    print("Shape of the dataframe:", df.shape)
    print("\nMissing values percentage:")
    print(df.isnull().sum() / len(df) * 100)

    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_cheater', data=df)
    plt.title('Distribution of is_cheater')
    plt.show()

    # Plotting distributions of some features
    features_to_plot = [
        'kill_death_ratio', 'headshot_percentage', 'accuracy_score',
        'spray_control_score', 'reports_received', 'account_age_days'
    ]
    
    for feature in features_to_plot:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=50)
        plt.title(f'Distribution of {feature}')
        plt.show()

if __name__ == '__main__':
    try:
        df = pd.read_csv('task1/train.csv')
        run_eda(df)
    except FileNotFoundError:
        print("Make sure 'task1/train.csv' is available.")