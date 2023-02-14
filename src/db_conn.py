#!/usr/bin/python3

import os
import toml
import pandas as pd
import psycopg2 as pg
from typing import List
from sqlalchemy import create_engine

env = toml.load("env.toml")


class DatabaseConn:
    def __init__(self, connector=env["db"]["DB_CONNECTOR"]) -> None:
        """Establish database connection based on the server settings in config.ini

        Args:
            log_file (str, optional): The name or path to the file where error will be logged. Defaults to ERROR_LOG path in config.ini
            connection (str, optional): The database connector, Default to DB_CONNECTOR in config.ini
        """
        # Load connection variables from config.ini
        self.DB_HOST = env["db"]["DB_HOST"]
        self.DB_PORT = env["db"]["DB_PORT"]
        self.DB_DATABASE = env["db"]["DB_DATABASE"]
        self.DB_USERNAME = env["db"]["DB_USERNAME"]
        self.DB_PASSWORD = env["db"]["DB_PASSWORD"]
        self.connector = connector

    def __alchemy(self):
        return create_engine(
            f"postgresql+psycopg2://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}/{self.DB_DATABASE}"
        )

    def __pg(self):
        # Establish a connection with the db
        try:
            conn = pg.connect(
                host=self.DB_HOST,
                dbname=self.DB_DATABASE,
                user=self.DB_USERNAME,
                password=self.DB_PASSWORD,
                port=self.DB_PORT,
            )
            return conn
        except Exception as e:
            print(e)

    def connect(self):
        conn = None
        if self.connector.lower() == "alchemy":
            conn = self.__alchemy()
        elif self.connector.lower() == "pg":
            conn = self.__pg()
        elif not self.connector.lower() in env["db"]["DB_CONNECTOR"]:
            raise TypeError(
                "Invalid connector, expects one of these " + env["db"]["DB_CONNECTOR"]
            )
        return conn

    def export_to_db(self, files: List, schema=env["db"]["DB_DEFAULT_SCHEMA"]):
        """Exports files to database

        Args:
            files (List): A list of filepaths
            schema (_type_, optional): _description_. Defaults to ccfd_staging.
        """
        try:

            conn = DatabaseConn(connector="alchemy")
            engine = conn.connect()
            for file in files:
                # Get the table name
                table_name = os.path.split(file)[-1].split(".")[0]

                data = pd.read_csv(file, delimiter=",", index_col=None).reset_index(
                    drop=True
                )
                data.to_sql(
                    name=table_name,
                    con=engine,
                    schema=schema,
                    if_exists="replace",
                    method="multi",
                )
                print(f"{table_name} records seeded successfuly")
        except Exception as e:
            print(e)
