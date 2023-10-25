## Train and Explain

These scripts pull snowflake credentials from `~/.snowsql`.

```
poetry install
poetry shell
python steps/1_load_data.py
python steps/2_train_a_model.py --model-version 1
```

Launch Streamlit:
```
streamlit run Home.py
```