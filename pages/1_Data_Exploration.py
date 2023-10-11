import streamlit as st

import snowflake.ml.modeling.preprocessing as snowml
from snowflake_ml import get_cache_session
import altair as alt

session = get_cache_session()

st.title("Data Preprocessing")

housing_df = session.table("HOUSING")
st.write("First, let's look at the head of our data")
with st.echo():
    df = housing_df.sample(n=1000).to_pandas()
    st.write(df.head(10))

quantitative_columns = df.columns[df.dtypes == "float64"].to_list()
categorical_columns = df.columns[df.dtypes == "object"].to_list()


st.subheader("Quantitative Columns")
dropdown = alt.binding_select(options=quantitative_columns, name="X-axis column ")
xcol_param = alt.param(value=quantitative_columns[0], bind=dropdown)
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("x:Q", bin=True).title(""),
        y="count()",
    )
    .transform_calculate(x=f"datum[{xcol_param.name}]")
    .add_params(xcol_param)
)

st.altair_chart(chart)

st.subheader("Categorical Columns")
dropdown = alt.binding_select(options=categorical_columns, name="X-axis column ")
xcol_param = alt.param(value=categorical_columns[0], bind=dropdown)
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("x:N"),
        y="count()",
    )
    .transform_calculate(x=f"datum[{xcol_param.name}]")
    .add_params(xcol_param)
)

st.altair_chart(chart)

st.write("Let's convert the Categorical Columns to Ordinal")

with st.echo():
    ordinal_encoder = snowml.OrdinalEncoder(
        input_cols=categorical_columns,
        output_cols=[c + "_ORD" for c in categorical_columns],
        drop_input_cols=True,
    )
    ordinal_encoder.fit(housing_df)
    st.write(ordinal_encoder.transform(housing_df.limit(10)))
