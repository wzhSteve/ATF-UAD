# ATF-UAD
This repository supplements our paper "An Adversarial Time-Frequency Reconstruction Network for Unsupervised Anomaly Detection"

## Installation
This code needs Python-3.7 and pytorch 1.8.1 or higher.
```bash
pip3 install -r requirements.txt
```

## Dataset Preprocessing
We have preprocessed all datasets and the link of them is shown as following, meanwhile we offer the checkpoints of all dataset to help you reproduce the results.
```bash
https://drive.google.com/file/d/1C3H9M0NdR3DViljjPzK889n6_FEHb4qr/view?usp=share_link
https://drive.google.com/file/d/19jNOoMbLSzAJjbUBrCE6V39oUFep0mkL/view?usp=share_link
```

## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset> --<process>
```
where `<model>` can be either of 'ATF-UAD' and other baselines. `<dataset>` can be one of 'SMAP', 'PSM', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR' and 'NAB. `<process>` can be `test` to reproduce the result based on the checkpoints and `retrain` to retrain the models.
