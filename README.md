# Beyond Averaging: A Window Wasserstein-based Aggregation for Ensemble Change Point Detection

Clean code and README are upcoming!

## Dependancies
```pip install -r requirements.txt```

## Datasets
All the data samples are available [online](https://disk.yandex.ru/d/_PQyni3AhyLu5g). Once loaded, put them into the ```/data``` folder.

## Pretrained models
All the pretrained models for ensembles are available [online](https://disk.yandex.ru/d/5iHHTOQDoIZhBQ). Once loaded, put them into the ```/saved_models``` folder.

## Experiments
* ```/scripts/train_single_models.py``` - script for training base models from scratch
* ```main.py``` - script for standard complete evaluation of a pretrained ensemble
* ```main_thresholds.py``` - script for evaluation of the proposed aggregation procedure with different threshold numbers
