# HDCL: Heterogeneous Debiasing  Contrastive Learning for Recommendation

## Requirements
- python==3.9.20
- pytorch==2.1.0
- dgl==2.4.0
- cuda==118

### Others
```python
pip install -r requirement.txt
```

## Running on Yelp, DoubanBook, Yelp, and DoubanMovie Datasets
```python
python main_HDCL.py --dataset Yelp --cl_rate 0.02  --lr 0.0005 --gpu 0 --batch 1024 --num_clusters 102 --cluster_level 2 --lambda_H 0.002 --lambda_T 3.0 --ts 0.5 --beta 1.0  --head_persent 85
python main_HDCL.py --dataset DoubanBook --cl_rate 0.1 --lr 0.0005 --gpu 0 --batch 1024 --num_clusters 102 --cluster_level 2 --lambda_H 0.001 --lambda_T 0.01 --ts 0.01 --beta 0.6  --head_persent 80
python main_HDCL.py --dataset DoubanMovie --cl_rate 0.08  --lr 0.0002 --gpu 0 --batch 1024 --num_clusters 102 --cluster_level 2 --lambda_H 0.006 --lambda_T 0.4 --ts 0.5 --beta 1.0  --head_persent 75
```








