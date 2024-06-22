Official implementation of "ClavaDDPM:Multi-relational Data Synthesis with Cluster-guided Diffusion Models" https://arxiv.org/pdf/2405.17724

To run a train-sampling pipeline:
```
python complex_pipeline --config_path configs/movie_lens.json
```

To use customized datasets:
See `complex_data/california` as an example. 
1. Save all tables as `.csv` files. All id columns should be named `<column_name>_id`.
2. Create a `dataset_meta.json`, in which `tables` should be manually created to specify all foreign key relationships in a multi-table dataset.
3. Create a `relation_order` in `dataset_meta.json`, which specifies the topological order of the multi-table dataset. The function `topological_sort` in `preprocess_utils.py` helps create it.
4. Create a domain file for each table, id columns excluded.
