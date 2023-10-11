from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
import streamlit as st


def get_session():
    params = connection_params.SnowflakeLoginOptions("aws")
    params["role"] = "MLE_ARCHITECTS"
    params["database"] = "SANDBOX"
    params["schema"] = "PHANSEN_WEEKLY_ML"
    session = Session.builder.configs(params).create()
    session.sql_simplifier_enabled = True
    return session


@st.cache_resource
def get_cache_session():
    return get_session()
