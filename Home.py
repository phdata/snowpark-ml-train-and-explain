import streamlit as st
from snowflake.snowpark.version import VERSION

# Snowpark ML
from snowflake_ml import get_session

st.set_page_config(
    page_title="Snowpark ML Demo",
    # layout="wide",
)

st.title("Train and Explain Using Snowpark ML")

st.write(
    """
This is a demonstration of the Snowpark ML Python library.
1. Explore the features available in the data.
2. Create a preprocessing pipeline.
3. Train a model using the Snowflake ML library.
4. Use Snowflake Model Registry to store artifacts and metrics.
5. Use Streamlit to visualize SHAP values to understand the model's behavior.
"""
)

session = get_session()


snowflake_environment = session.sql(
    "SELECT current_user(), current_version()"
).collect()
snowpark_version = VERSION

# Current Environment Details
print("\nConnection Established with the following parameters:")
print("User                        : {}".format(snowflake_environment[0][0]))
print("Role                        : {}".format(session.get_current_role()))
print("Database                    : {}".format(session.get_current_database()))
print("Schema                      : {}".format(session.get_current_schema()))
print("Warehouse                   : {}".format(session.get_current_warehouse()))
print("Snowflake version           : {}".format(snowflake_environment[0][1]))
print(
    "Snowpark for Python version : {}.{}.{}".format(
        snowpark_version[0], snowpark_version[1], snowpark_version[2]
    )
)
