import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Define file paths
    input_file = "data/train.csv"
    train_output_file = "data/train_cleaned.csv"
    val_output_file = "data/val_cleaned.csv"

    # Read only the required columns
    df = pd.read_csv(input_file, usecols=["id", "glasses"])

    # Split the dataset into 80% train and 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the train and test data into separate CSV files
    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(val_output_file, index=False)

    # Print confirmation messages
    print(f"Train data saved to {train_output_file}")
    print(f"Val data saved to {val_output_file}")


# Run the main function if the script is executed
if __name__ == "__main__":
    main()
