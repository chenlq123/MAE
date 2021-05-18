# MAE

Our paper [Multi-Attribute Enhancement Network for Person Search](https://arxiv.org/ftp/arxiv/papers/2102/2102.07968.pdf) has been accepted by IJCNN2021. Our code is stored in this repository.

## Preparation
  
  
  1. Clone this repo:
  
   ```bash
  https://github.com/chenlq123/MAE.git && cd MAE
   ```
  
  
  2. Requirements
  
  Python 3.6 with all requirements.txt(except for the first line) dependencies installed. To install run:
  
   ```bash
  pip install -r requirements.txt
   ```
  
  
  
  3. Download the [Trained Model](https://github.com/chenlq123/MAE/releases/download/v1.0/pre_train.zip) from the Release.
  
  
  
  4. Download the dataset of CUHK-SYSU and PRW.
  
  
  
  5. Download the [Attribute Label](https://github.com/chenlq123/MAE/releases/download/a1.0/Attribute.Label.zip) from the Release. And put them in dataset, respectively.
  
  
  6. Before Training
  
  You need to modify the *path* in the *lib/datasets/__init__.py* and *lib/model/faster_rcnn_mae.py*.
  
  
  
  7. Test
  
  
  For CUHK-SYSU
  
   ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/test_MAE.py -p pre_logs/cuhk_sysu/
   ```
  
  
  For PRW
   ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/test_MAE.py -p pre_logs/prw/  --dataset PRW
   ```
  
  
  8. Train
   ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train_MAE.py --debug --lr_warm_up -p ./logs/ --batch_size 2 --nw 2 --w_RCNN_loss_bbox 10.0 --epochs 22 --lr 0.003 --lr_decay_step 8
   ```
  
  
  
  
  
## Citation

```latex
@inproceedings{chen2020norm,
  title={Multi-Attribute Enhancement Network for Person Search},
  author={Lequan Chen, Wei Xie, Zhigang Tu, Jinglei Guo, Yaping Tao, Xinming Wang},
  booktitle={IJCNN},
  year={2021}
}
```
