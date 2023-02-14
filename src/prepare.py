#!/usr/bin/env python3

from datetime import datetime as dt
from zipfile import ZipFile
import pandas as pd
import os


# Unzip file
with ZipFile("data/raw/creditcard.csv.zip") as zip:
    zip.extractall("./data/raw")

# read extracted data using pandas
raw_data_path = "data/raw/creditcard.csv"
data = pd.read_csv(raw_data_path)
data["ingestion_date"] = dt.strftime(dt.now(), "%Y-%m-%d:%H:%M")


# Drop duplicate values
is_duplicated = True if data.duplicated().sum() > 0 else False
if is_duplicated:
    new_data = data.drop_duplicates()


# Drops null values
if data.isna().sum().sum():
    data.dropna(inplace=True)

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
data.to_csv("data/prepared/prepared.csv", index=False)
