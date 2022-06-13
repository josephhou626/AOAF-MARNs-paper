# AOAF-MARNs-paper
Image Denoising Using Adaptive and Overlapped Average Filtering and Mixed-Pooling Attention Reﬁnement Networks paper

[![paper](https://img.shields.io/badge/mathematics-paper-green)](https://www.mdpi.com/2227-7390/9/10/1130)

## Network Architecture

<img src="./figures/Architecture.jpg" width = "1000" div align=center />

## Environment
- Windows 10 
- GeForce RTX 3090 GPU
- python3.8.6
- torch=1.8.1
- torchvision=0.9.1

## Installation

1. Install virtual environment:
	```shell
	virtualenv -p python3 exp3 # establish
	.\exp3\Scripts\activate # activate 
	```

2. Clone this repo:
	```shell
	git clone https://github.com/josephhou626/AOAF-MARNs-paper.git # clone
	```

3. Install torch and torchvision:
	```shell
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
	```

4. Install dependencies:
   ```shell
   cd MARNs
   pip install -r requirements.txt
   ```


## Dataset

我們採用DIV2k 的資料集


## AOAF

使用matlab執行AOAF/main.m

把結果放入
   ```
   AOAF_results/train/input
   AOAF_results/test/input
   ```

## MARNs Train

```
python main.py --mode train 
```


## MARNs Test

```
python main.py --mode test --load_model_path exp/MARNs/model/model_last
```

## Results
**Quantitative Evaluations**
<img src="./figures/table.jpg" width = "1000" div align=center />

**Qualitative Evaluations**
<img src="./figures/vis_result.jpg" width = "1000" div align=center />

## Evaluation

執行measure/measure_main.m 計算PSNR和SSIM分數。

## Reference

- "[SPNet](https://github.com/Andrew-Qibin/SPNet)"

