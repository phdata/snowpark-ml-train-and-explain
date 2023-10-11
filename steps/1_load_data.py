from snowflake_ml import get_session
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

session = get_session()
session.sql(f"CREATE OR REPLACE SCHEMA {session.get_current_schema()}")

housing = pd.read_csv("data/housing.csv")
housing.columns = housing.columns.str.upper()
print(housing.shape)
table = session.write_pandas(
    housing,
    "HOUSING",
    database="SANDBOX",
    schema="PHANSEN_WEEKLY_ML",
    auto_create_table=True,
    overwrite=True,
)

print(table.count())
print(table.describe().show())
