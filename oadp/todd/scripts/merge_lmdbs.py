import argparse

import lmdb


def parse_args():
    parser = argparse.ArgumentParser(description='Merge LMDBs')
    parser.add_argument('target_filepath')
    parser.add_argument('--source-filepaths', type=str, nargs='+')
    parser.add_argument('--dbs', type=str, nargs='+')
    args = parser.parse_args()
    return args


def merge_txns(target_txn: lmdb.Transaction, source_txn: lmdb.Transaction):
    with source_txn.cursor() as cur:
        for k, v in cur:
            target_txn.put(k, v)


def merge_dbs(
    target_env: lmdb.Environment,
    source_env: lmdb.Environment,
    dbs,
):
    for db in dbs:
        target_db = target_env.open_db(db.encode())
        source_db = source_env.open_db(db.encode())
        with target_env.begin(
            target_db,
            write=True,
        ) as target_txn, source_env.begin(source_db) as source_txn:
            merge_txns(target_txn, source_txn)


def merge_envs(
    target_env: lmdb.Environment,
    source_filepaths,
    dbs,
):
    for source_filepath in source_filepaths:
        with lmdb.open(
            source_filepath,
            readonly=True,
            max_dbs=len(dbs),
        ) as source_env:
            merge_dbs(target_env, source_env, dbs)


def main():
    args = parse_args()
    with lmdb.open(
        args.target_filepath,
        map_size=10 * 2**30,  # 10 GB
        readonly=False,
        max_dbs=len(args.dbs),
    ) as target_env:
        merge_envs(target_env, args.source_filepaths, args.dbs)


if __name__ == '__main__':
    main()
