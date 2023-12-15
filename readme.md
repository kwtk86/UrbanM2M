# UrbanM2M

A ConvLSTM-based model for simulating future urban expansion.
If you find this work useful, please cite our work at [UrbanM2M-IJGIS]()

基于ConvLSTM神经网络的城市扩张模拟模型。
如果该模型对你有帮助，请引用我们的论文 [UrbanM2M-IJGIS]()。



## Use UrbanM2M-tester in OpenGMS platform 在OpenGMS上使用UrbanM2M (recommended)
It is recommmend to use UrbanM2M in OpenGMS to simulate future urban expansion in your research area,
because OpenGMS provides users with free computation resources.


As UrbanM2M is regional-transferable, 
you can simulate future urban expansion using various types of trained model directly,
without costing time to train your own model.

However, only testing UrbanM2M is allowed in OpenGMS. 
If you want to **train your own model**, please see the next section for reference.

我们推荐在OpenGMS上使用UrbanM2M进行城市扩张模拟。OpenGMS是南京师范大学开发的一个在线地理模拟平台，
用户可以在平台上上免费的使用算力调用UrbanM2M，而不必纠结于电脑配置，且不用自己配置环境。

在OpenGMS，用户可以使用提前训练好的模型进行城市扩张模拟，而不必花费时间自己训练模型。

但在OpenGMS，用户仅可以调用已经训练好的模型进行模拟而不能自己训练模型。如果希望**训练模型**，请参考下一节。

See [OpenGMS-UrbanM2M]() for detail.

## Use UrbanM2M with Gradio GUI in your device 

You can also deploy and use UrbanM2M locally.

See [UrbanM2M用户手册中文版]() or [UrbanM2M user manual-en]() for detail


[//]: # ()
[//]: # (## Code with UrbanM2M package)

[//]: # ()
[//]: # ()
[//]: # (### Development environment configuration)

[//]: # ()
[//]: # (**Note**: The following process has been validated in both Windows &#40;for all the models&#41; and Centos7 &#40;for UrbanM2M&#41;, but haven't been validated in MacOS yet.)

[//]: # ()
[//]: # (**First**, install a [Miniconda]&#40;https://docs.conda.io/en/latest/miniconda.html#installing&#41; or [Anaconda]&#40;https://www.anaconda.com/&#41; package manager.)

[//]: # ()
[//]: # (**Second**, run the following commands in cmd or bash to create a new virtual environment for UrbanM2M.)

[//]: # ()
[//]: # (``` bash)

[//]: # (conda create -n urbanm2m python==3.10.0)

[//]: # (```)

[//]: # ()
[//]: # (**Third**, install PyTorch)

[//]: # ()
[//]: # (``` bash)

[//]: # (conda activate urbanm2m)

[//]: # (conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch)

[//]: # (```)

[//]: # ()
[//]: # (**Fourth**, install GDAL-Python&#40;version>=3.0.0&#41;)

[//]: # ()
[//]: # (If you are using Linux, run ```conda install gdal``` directly to install it.)

[//]: # ()
[//]: # (If you are using Windows, it is better to install GDAL-Python using the .whl file in the UrbanM2M folder.)

[//]: # (``` bash)

[//]: # (cd UrbanM2M)

[//]: # (pip install GDAL-3.4.3-cp310-cp310-win_amd64.whl)

[//]: # (``` )

[//]: # ()
[//]: # ()
[//]: # (**Finally**, run ```pip install requirements.txt``` to install other dependency packages.)

[//]: # ()
[//]: # (**Note**: please ensure all the procedures above are run in the **urbanm2m** Virtual Environment)

[//]: # ()
[//]: # (### UrbanM2M model implementation)

[//]: # ()
[//]: # ()
[//]: # (#### **Training your model**)

[//]: # (cd ./m2mCode)

[//]: # (Run ```train_gui.py``` using an IDE directly or run the following command in cmd or bash &#40;**recommended**&#41;.)

[//]: # (```bash)

[//]: # (python train_gui.py --start_year 2000 --in_len 6 --out_len 6 --data_dir ../data-yrd --spa_vars county.tif town.tif slope.tif --batch_size 8 --lr 0.00005 --sample_count 5000 --val_prop 0.15)

[//]: # (```)

[//]: # ()
[//]: # (**Note**: the parameters are modifiable. Especially, check your GPU memory to set ```batch_size```. Per batch size needs about 1.3GB GPU memory.)

[//]: # ()
[//]: # (**Note**: it is recommended to end training after 20 epochs. The trained models will be storaged in ```trained_models``` folder.)

[//]: # ()
[//]: # (#### **Testing your model**)

[//]: # ()
[//]: # (After finishing training, modify the parameters and run ```test_gui.py``` using an IDE directly or run the following command in cmd or bash &#40;**recommended**&#41;.)

[//]: # ()
[//]: # (``` bash)

[//]: # (python test_gui.py --start_year 2006 --in_len 6 --out_len 6 --height 64 --block_step 38 --edge_width 4 --spa_vars slope\|town\|county --model_type gba --region gba --data_root_dir ../data-gisa-gba --log_file ./mylog/gba.csv --model_path ./trained_models/gba-fs5-e23-p.pth --run_model True --numworkers 0 --batch_size 100)

[//]: # (```)

[//]: # ()
[//]: # (Aftering finish testing, results can be found in ```data-gisa-gba/sim``` folder.)

[//]: # ()
[//]: # (**Note**: the parameter model_path must be modified according to name of your trained model.  )

[//]: # ()
[//]: # (**Note**: check your GPU memory to set ```batch_size```. Per batch size needs about 40MB GPU memory when testing.)

[//]: # ()
[//]: # (**Note**: a trained model has been prepared in the ```trained_models``` folder, you can directly test this model without training.)

[//]: # ()
