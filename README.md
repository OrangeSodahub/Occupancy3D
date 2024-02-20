### Instructions for development

#### Install

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**1. Create a conda virtual environment and activate it.**
```shell
conda create -n surroundocc python=3.7 -y
conda activate surroundocc
```

**2. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0
```

**5. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**6. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**7. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
```

**8. Install Chamfer Distance.**
```shell
cd SurroundOcc/extensions/chamfer_dist
python setup.py install --user
```

**9. Prepare pretrained models.**
```shell
cd SurroundOcc 
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

#### Data

**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```


**2. Download the generated [train](https://cloud.tsinghua.edu.cn/f/ebbed36c37b248149192/?dl=1)/[val](https://cloud.tsinghua.edu.cn/f/b3f169f4db034764bb87/?dl=1) pickle files and put them in data.**

**3. Download our generated dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) and put and unzip it in data**
| Subset | Tsinghua Cloud| Size |
| :---: | :---: | :---: |
| train | [link](https://cloud.tsinghua.edu.cn/f/f021006560b54bc78349/?dl=1) | 4.3G |
| val | [link](https://cloud.tsinghua.edu.cn/f/290276f4a4024896b733/?dl=1) | 627M |

**Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   ├── nuscenes_occ/
│   ├── nuscenes_infos_train.pkl
│   ├── nuscenes_infos_val.pkl

```

***4. (Optional) We also provide the code to generate occupancy on nuScenes, which needs LiDAR point semantic labels [HERE](https://www.nuscenes.org/download). Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
|   |   ├── lidarseg
|   |   |   ├── v1.0-test
|   |   |   ├── v1.0-trainval
|   |   |   ├── v1.0-mini
```

You can generate train/val split of nuScenes from 850 sequences. 

```
cd $Home/tools/generate_occupancy_nuscenes
python generate_occupancy_nuscenes.py --config_path ./config.yaml --label_mapping ./nuscenes.yaml --split [train/val] --save_path [your/save/path] 
```

#### Train and Test

Train SurroundOcc with 8 RTX3090 GPUs 
```
./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc.py 8  ./work_dirs/surroundocc
```

Eval SurroundOcc with 8 RTX3090 GPUs
```
./tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ./path/to/ckpts.pth 8
```
You can substitute surroundocc.py with surroundocc_nosemantic.py for 3D scene reconstruction task.

Visualize occupancy predictions:

First, you need to generate prediction results. Here we use whole validation set as an example.
```
cp ./data/nuscenes_infos_val.pkl ./data/infos_inference.pkl
./tools/dist_inference.sh ./projects/configs/surroundocc/surroundocc_inference.py ./path/to/ckpts.pth 8
```
You will get prediction results in './visual_dir'. You can directly use meshlab to visualize .ply files or run visual.py to visualize raw .npy files with mayavi:
```
cd ./tools
python visual.py $npy_path$
```
