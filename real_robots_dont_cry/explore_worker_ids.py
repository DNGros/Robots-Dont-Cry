from pathlib import Path

import pandas as pd

from util.sampling import deterministic_hash

cur_file = Path(__file__).parent.absolute()

secret_salt = (cur_file / "secret_worker_salt_val.txt").read_text()


def hash_worker_id(worker_id):
    return deterministic_hash((worker_id, secret_salt), seed=1, digest_bytes=8)


def main():
    transaction_df = pd.read_csv(cur_file / "responses/Transactions_2022-03-11_to_2022-03-29.csv")
    transaction_df = transaction_df[transaction_df['Recipient ID'] != 'Mechanical Turk']
    print(transaction_df.columns)
    print(transaction_df.head())
    print(transaction_df.shape)
    print(transaction_df.groupby('Recipient ID').count().sort_values('Transaction ID', ascending=False).head(30))


if __name__ == "__main__":
    main()
