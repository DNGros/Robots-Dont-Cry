from pathlib import Path
import pandas as pd
cur_file = Path(__file__).parent.absolute()


def get_transactions_df():
    transaction_df = pd.read_csv(cur_file / "responses/Transactions_2022-03-01_to_2022-03-31.csv")
    transaction_df = transaction_df[transaction_df['Recipient ID'] != 'Mechanical Turk']
    first_dates = transaction_df.groupby('Recipient ID')['Date Initiated'].transform('min')
    transaction_df['calc_earliest_date'] = first_dates
    transaction_df['calc_is_earliest'] = transaction_df['Date Initiated'] == transaction_df['calc_earliest_date']
    assert len(transaction_df['Assignment ID'].unique()) == len(transaction_df)
    return transaction_df


def main():
    transaction_df = get_transactions_df()
    print(transaction_df.columns)
    print(transaction_df.head())
    print(transaction_df.shape)


if __name__ == "__main__":
    main()
