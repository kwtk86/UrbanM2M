# UrbanM2M

## Development environment configuration

**Note**: The following process has been validated in both Windows (for all the models) and Centos7 (for UrbanM2M), but haven't been validated in MacOS yet.

**First**, install a [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [Anaconda](https://www.anaconda.com/) package manager.

**Second**, run the following commands in cmd or bash to create a new virtual environment for UrbanM2M.

``` bash
conda create -n urbanm2m python==3.10.0
```

**Third**, install PyTorch

``` bash
conda activate urbanm2m
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
```

**Fourth**, install GDAL-Python(version>=3.0.0)

If you are using Linux, run ```conda install gdal``` directly to install it.

If you are using Windows, it is better to install GDAL-Python using the .whl file in the UrbanM2M folder.
``` bash
cd UrbanM2M
pip install GDAL-3.4.3-cp310-cp310-win_amd64.whl
``` 


**Finally**, run ```pip install requirements.txt``` to install other dependency packages.

**Note**: please ensure all the procedures above are run in the **urbanm2m** Virtual Environment

## UrbanM2M model implementation

### **Generating raster tiles**

``` bash
cd ./m2mCode
python split.py
```
Note that this process would be very slow if data is on HHD disk, and much faster if data is on SSD disk.

By running split.py, two folders storaging tiles will be generated in the ```data-gisa-gba``` folder

### **Training your model**
Run ```train.py``` using an IDE directly or run the following command in cmd or bash (**recommended**).
```bash
python train.py --start_year 2000 --in_len 6 --out_len 6 --tile_size 64 --block_dir ../data-gisa-gba/block64_64 --spa_vars slope\|town\|county --nlayers 2 --filter_size 5 --epochs 60 --batch_size 8 --lr 0.00005 --eta_decay 0.015 --sample_count 5000 --val_prop 0.25 --model_type gba
```

**Note**: the parameters are modifiable. Especially, check your GPU memory to set ```batch_size```. Per batch size needs about 1.3GB GPU memory.

**Note**: it is recommended to end training after 20 epochs. The trained models will be storaged in ```trained_models``` folder.

### **Testing your model**

After finishing training, modify the parameters and run ```train.py``` using an IDE directly or run the following command in cmd or bash (**recommended**).

``` bash
python test.py --start_year 2006 --in_len 6 --out_len 6 --height 64 --block_step 38 --edge_width 4 --spa_vars slope\|town\|county --model_type gba --region gba --data_root_dir ../data-gisa-gba --log_file ./mylog/gba.csv --model_path ./trained_models/gba-fs5-e23-p.pth --run_model True --numworkers 0 --batch_size 100
```

Aftering finish testing, results can be found in ```data-gisa-gba/sim``` folder.

**Note**: the parameter model_path must be modified according to name of your trained model.  

**Note**: check your GPU memory to set ```batch_size```. Per batch size needs about 40MB GPU memory when testing.

**Note**: a trained model has been prepared in the ```trained_models``` folder, you can directly test this model without training.

**LSTM-CA implementation**