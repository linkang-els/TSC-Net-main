
TSC-Net for Weakly supervised timing action localization implementation code (pytorch version)

# Dependencies
* Create the conda environment as what I used.

``` 
conda create -n TFEDCN python=3.6

conda activate TFEDCN

pip install -r requirements.txt

pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm==4.41.1
``` 

# THUMOS'14 Dataset
The feature for THUMOS'14 Dataset can be downloaded here. The annotations are included with this package.

* [Google Drive](https://drive.google.com/file/d/1YLmbv_6bd696iN_5UdknP_gadNTytbSM/view?usp=share_link)

# Training
* Run the train scripts:
``` 
python main.py --max-seqlen 320 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --path-dataset ./dataset --num-class 20 --use-model TSCNET_V1 --max-iter 3010 --dataset SampleDataset --weight_decay 0.001 --model-name TSCNET_V1


``` 

# References
We referenced the following repos for the code:
* [ActivityNet](https://github.com/activitynet/ActivityNet)
* [MM2021-CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)

