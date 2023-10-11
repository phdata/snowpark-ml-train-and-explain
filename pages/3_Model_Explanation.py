import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import shap
import streamlit as st
from matplotlib.cm import get_cmap
from snowflake.ml._internal.utils import identifier
from snowflake.ml.registry import model_registry
from sklearn.metrics import mean_absolute_percentage_error

from snowflake_ml import get_cache_session

session = get_cache_session()

st.title("Explore SHAP Values for Model")

st.write(
    "Let's get our registered model and explain the predictions of our model using SHAP"
)

housing_df = session.table("HOUSING")
target = "MEDIAN_HOUSE_VALUE"
df = housing_df.limit(1).to_pandas()
quantitative_columns = df.columns[df.dtypes == "float64"].to_list()
categorical_columns = df.columns[df.dtypes == "object"].to_list()
quantitative_columns.remove(target)
oe_output_cols = [c + "_ORD" for c in categorical_columns]
columns = oe_output_cols + quantitative_columns

st.header("Retrieve Model from Registry")

db = identifier._get_unescaped_name(session.get_current_database())
schema = identifier._get_unescaped_name(session.get_current_schema())


@st.cache_resource
def get_registry():
    return model_registry.ModelRegistry(
        session=session,
        database_name=db,
        schema_name=schema,
        create_if_not_exists=True,
    )


st.write(
    get_registry()
    .list_models()
    .select("NAME", "TYPE", "VERSION", "URI", "METRICS")
    .to_pandas()
)

model_version = st.number_input("Model Version", value=1, step=1)
model_name = st.text_input("Model Name", value="housing_model")


@st.cache_resource
def get_model(model_version):
    return model_registry.ModelReference(
        registry=get_registry(), model_name=model_name, model_version=model_version
    )


model_ref = get_model(model_version)

pipeline = model_ref.load_model()


@st.cache_resource
def train_test_split():
    df_train, df_test = housing_df.random_split([0.8, 0.2], seed=42)
    return df_train, df_test


df_train, df_test = train_test_split()
model_deployment_name = f"{model_name}{model_version}_UDF"


@st.cache_data
def predict(_df, name):
    result_sdf = model_ref.predict(deployment_name=model_deployment_name, data=_df)
    return result_sdf.to_pandas()


@st.cache_data
def get_mape():
    return model_ref.get_metrics()["mean_abs_pct_err"]


mape = get_mape()

df_test_pred = predict(df_test, "test")

ordinal_encoder = pipeline.steps[0][1].to_sklearn()
model = pipeline.steps[1][1].to_xgboost()
model_input_cols = pipeline.steps[1][1].input_cols

selection = st.radio(
    "Select Explanation Type",
    ["Summary", "What-If", "Partial Dependence", "Shap Location"],
    horizontal=True,
)


@st.cache_data
def explain(df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df[model_input_cols])
    return explainer, shap_values


explainer, shap_values = explain(df_test_pred)

if selection == "What-If":
    with st.expander("Ordinal Encoder Categories"):
        st.write("This table shows the categories for each ordinal encoded column")
        st.write(pd.DataFrame(dict(zip(oe_output_cols, ordinal_encoder.categories_))))

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    if st.button("Get Random Row"):
        st.session_state.pop("row")
    row = st.session_state.get("row")
    if row is None:
        row = df_test_pred.sample(1)[model_input_cols].T.copy()
        row.columns = ["Example"]
    st.session_state["row"] = row

    @st.cache_resource
    def waterfall(rows):
        explainer_row = explainer(rows)
        fig, _ = plt.subplots()
        shap.plots.decision(
            explainer_row.base_values[0],
            explainer_row.values,
            explainer_row.data,
            explainer_row.feature_names,
            show=False,
            highlight=1,
            legend_labels=["Example", "What-If"],
            legend_location="lower right",
        )
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("An example row of data")
        st.dataframe(row)
    with col2:
        st.write("Change values here to compare")
        row_whatif = st.data_editor(row, num_rows="fixed", key="whatif")

    rows = pd.concat([row, row_whatif], axis=1).T
    waterfall(rows)

if selection == "Summary":
    st.write("Summary of SHAP Values")
    fig, _ = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)

    chart = (
        alt.Chart(df_test_pred)
        .mark_circle()
        .encode(
            x=alt.X(target, scale=alt.Scale(zero=False)),
            y=alt.Y(target + "_PRED", scale=alt.Scale(zero=False)),
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.metric(
        "MAPE",
        f"{mean_absolute_percentage_error(df_test_pred[target], df_test_pred[target + '_PRED']):.4f}",
    )

if selection == "Partial Dependence":
    st.header("Partial Dependence Plots")
    min_shap = shap_values.min(None).values
    max_shap = shap_values.max(None).values
    y_min = min_shap - (max_shap - min_shap) * 0.1
    y_max = max_shap + (max_shap - min_shap) * 0.1
    for col in shap_values.abs.mean(0).argsort.values[::-1]:
        st.subheader(shap_values.feature_names[col])
        fig, ax = plt.subplots()
        shap.plots.scatter(shap_values[:, col], ax=ax, ymin=y_min, ymax=y_max)
        st.pyplot(fig)

if selection == "Shap Location":
    st.header("SHAP Location")

    # Make a pydeck plot of shap values based on Lat/Lon
    df_train_pred = predict(df_train, "train")
    df_pred_all = pd.concat([df_train_pred, df_test_pred], axis=0).sample(n=10_000)

    explainer_all, shap_values_all = explain(df_pred_all)

    def plot_shap_map(data, shap_values, name):
        data = data.copy()

        # Normalize SHAP values
        shap_min = np.percentile(shap_values, 5)
        shap_max = np.percentile(shap_values, 95)
        shap_loc_normalized = (shap_values - shap_min) / (shap_max - shap_min)

        data["SHAP_LOC"] = shap_loc_normalized
        data["SHAP_VALUES"] = shap_values
        data[target + "_f"] = data[target].apply(lambda x: f"${x:,.0f}")
        data[target + "_PRED_f"] = data[target + "_PRED"].apply(lambda x: f"${x:,.0f}")

        # Create a color map
        cm = get_cmap("coolwarm")

        # Calculate fill_color values
        data["fill_color"] = [cm(value) for value in data["SHAP_LOC"]]
        print(data.fill_color.iloc[0])
        data["fill_color"] = data["fill_color"].apply(
            lambda x: (255 * x[0], 255 * x[1], 255 * x[2], 255 * x[3])
        )

        st.write(data.head())
        st.subheader(name)
        tooltip = (
            f"Pred: {{{target}_PRED_f}}\n"
            f"Actual: {{{target}_f}}\n"
            "Contribution: {SHAP_VALUES}\n"
        )
        if name != "LOCATION":
            tooltip += f"{name}: {{{name}}}\n"

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=data["LATITUDE"].mean(),
                    longitude=data["LONGITUDE"].mean(),
                    zoom=5,
                    pitch=0,
                ),
                tooltip={"text": tooltip},
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=data,
                        pickable=True,
                        opacity=0.5,
                        stroked=False,
                        filled=True,
                        radius_scale=6,
                        line_width_min_pixels=1,
                        get_position=["LONGITUDE", "LATITUDE"],
                        get_radius=200,
                        get_fill_color="fill_color",
                        get_line_color=[0, 0, 0],
                    )
                ],
            )
        )

    st.write(df_test_pred.head())
    if "LATITUDE" in model_input_cols:
        plot_shap_map(
            df_pred_all,
            shap_values=shap_values_all[:, ["LATITUDE", "LONGITUDE"]]
            .sum(axis=1)
            .values,
            name="LOCATION",
        )

    for col in shap_values_all.abs.mean(0).argsort.values[::-1]:
        c = shap_values_all.feature_names[col]
        if c in ["LATITUDE", "LONGITUDE"]:
            continue
        plot_shap_map(
            df_pred_all,
            shap_values=shap_values_all[:, col].values,
            name=c,
        )
