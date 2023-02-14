#!/usr/bin/env python3

from db_conn import DatabaseConn
import pandas as pd
import toml

# load env variables
env = toml.load("env.toml")

# connect to db
db_conn = DatabaseConn()
engine = db_conn.connect()

data = pd.read_csv(
    "data/prepared/prepared.csv", delimiter=",", index_col=None
).reset_index(drop=True)

# backup data to database
try:
    data.to_sql(
        name="trx",
        con=engine,
        schema=env["db"]["DB_DEFAULT_SCHEMA"],
        if_exists="replace",
        method="multi",
    )
except Exception as e:
    print(e)
