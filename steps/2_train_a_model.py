import snowflake.ml.modeling.preprocessing as snowml
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error
from snowflake.ml.registry import model_registry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.exceptions.exceptions import SnowflakeMLException
from snowflake.connector.errors import DataError
import click


from snowflake_ml import get_session


@click.command()
@click.option("--model-version", default=1, help="Model Version")
def main(model_version):
    session = get_session()

    # Define input and output columns

    target = "MEDIAN_HOUSE_VALUE"
    quantitative_columns = [
        "MEDIAN_INCOME",
        "TOTAL_BEDROOMS",
        "POPULATION",
        "HOUSING_MEDIAN_AGE",
        "HOUSEHOLDS",
        "TOTAL_ROOMS",
    ]
    categorical_columns = ["OCEAN_PROXIMITY"]
    oe_output_cols = [c + "_ORD" for c in categorical_columns]

    # Create preprocessing step
    ordinal_encoder = snowml.OrdinalEncoder(
        input_cols=categorical_columns,
        output_cols=oe_output_cols,
    )
    # Create model step
    reg = XGBRegressor(
        n_estimators=50,
        input_cols=oe_output_cols + quantitative_columns,
        label_cols=[target],
        output_cols=[target + "_PRED"],
    )

    # Create Snowpark ML Pipeline
    pipeline = Pipeline(
        [
            ("ordinal_encoder", ordinal_encoder),
            ("model", reg),
        ]
    )

    # Train/Test Split and Fit
    housing_df = session.table("HOUSING")
    df_train, df_test = housing_df.random_split([0.8, 0.2], seed=42)
    pipeline.fit(df_train)

    # Calculate Metrics
    df_test_pred = pipeline.predict(df_test)
    mape = mean_absolute_percentage_error(
        df=df_test_pred,
        y_true_col_names=target,
        y_pred_col_names=target + "_PRED",
    )

    X = df_train.select(categorical_columns + quantitative_columns).limit(100)
    db = identifier._get_unescaped_name(session.get_current_database())
    schema = identifier._get_unescaped_name(session.get_current_schema())
    # Define model name and version
    model_name = "housing_model"

    # Create a registry and log the model and Metrics
    registry = model_registry.ModelRegistry(
        session=session, database_name=db, schema_name=schema, create_if_not_exists=True
    )

    try:
        registry.log_model(
            model_name=model_name,
            model_version=model_version,
            model=pipeline,
            sample_input_data=X,
            options={
                "embed_local_ml_library": True,
                "relax": True,
            },
        )

        # Add evaluation metric
        registry.set_metric(
            model_name=model_name,
            model_version=model_version,
            metric_name="mean_abs_pct_err",
            metric_value=mape,
        )
    except (SnowflakeMLException, DataError) as e:
        if f"Model {model_name}/{model_version} already exists" in str(e):
            print(
                f"Model {model_name}/{model_version} already exists, skipping registration"
            )

    print(
        registry.list_models()
        .select("NAME", "TYPE", "VERSION", "URI", "METRICS")
        .to_pandas()
    )

    model_deployment_name = f"{model_name}{model_version}_UDF"

    registry.deploy(
        model_name=model_name,
        model_version=model_version,
        deployment_name=model_deployment_name,
        target_method="predict",
        permanent=True,
        options={"relax_version": True},
    )
    print(registry.list_deployments(model_name, model_version).to_pandas())


if __name__ == "__main__":
    main()
