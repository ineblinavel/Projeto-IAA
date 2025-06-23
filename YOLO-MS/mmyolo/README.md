<div align="center">

  <a href="../README.md">
    <img src='../assets/logo.png' alt='ICLR2025_REALLOD_LOGO' width="250px"/><br/>
  </a>
  
  <h2 style="margin-right: 10px;"> MMYOLO version 
    <a href="../README.md" style="text-decoration: none; color: inherit; display: inline-block; vertical-align: middle;">
        <span style="font-size: 16px; margin-left: 5px; vertical-align: middle;">🔙</span>
    </a>
  </h2>
</div>

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🛠️ Dependencies and Installation 🔝](#-dependencies-and-installation-)
- [👼 Quick Demo 🔝](#-quick-demo-)
- [🏋️ Training 🔝](#-training-)
- [📊 Evaluation 🔝](#-evaluation-)
- [📦 Deployment 🔝](#-deployment-)

## 🛠️ Dependencies and Installation [🔝](#-table-of-contents)

> We provide a simple scrpit `install.sh` for installation, or refer to [install.md](../docs/install_mmyolo.md) for more details.

1. Clone and enter the repo.

   ```shell
   git clone https://github.com/FishAndWasabi/YOLO-MS.git
   cd YOLO-MS/mmyolo
   ```

2. Run `install.sh`.

   ```shell
   bash install.sh
   ```

3. Activate your environment!

   ```shell
   conda activate YOLO-MS-mmyolo
   ```

## 👼 Quick Demo [🔝](#-table-of-contents)

```shell
python demo/image_demo.py ${IMAGE_PATH} ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# for sam output
python demo/sam_demo.py ${IMAGE_PATH} ${CONFIG_FILE} ${CHECKPOINT_FILE} --sam_size ${SAM_MODEL_SIZE} --sam_model ${SAM_MODEL_PATH}
```

You could run `python demo/image_demo.py --help` to get detailed information of this scripts.

<details>
<summary> Detailed arguments </summary>

```
positional arguments:
  img                   Image path, include image file, dir and URL.
  config                Config file
  checkpoint            Checkpoint file

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR     Path to output file
  --device DEVICE       Device used for inference
  --show                Show the detection results
  --deploy              Switch model to deployment mode
  --tta                 Whether to use test time augmentation
  --score-thr SCORE_THR
                        Bbox score threshold
  --class-name CLASS_NAME [CLASS_NAME ...]
                        Only Save those classes if set
  --to-labelme          Output labelme style label file
  
  --sam_size            Default: vit_h, Optional: vit_l, vit_b
  --sam_model           Path of the sam model checkpoint
```

</details>

<table>
  <tbody>
    <tr>
        <td>
            <img src='demo/demo.jpg' alt='DEMO' width='500px'/>
        </td>
        <td>
            <img src='../assets/demo_output.jpg' alt='DEMO_OUTPUT' width='500px'/>
        </td>
    </tr>
    <tr>
        <td>
            <img src='demo/demo.jpg' alt='DEMO' width='500px'/>
        </td>
        <td>
            <img src='../assets/demo_sam.jpg' alt='DEMO_SAM_OUTPUT' width='500px'/>
        </td>
    </tr>
    </tbody>
</table>

## 🏋️ Training [🔝](#-table-of-contents)

### 1. Single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

### 2. Multi GPU

```shell
CUDA_VISIBLE_DEVICES=x bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

You could run `python tools/train.py --help` to get detailed information of this scripts.

<details>
<summary> Detailed arguments </summary>

```
positional arguments:
    config                train config file path

optional arguments:
    -h, --help            show this help message and exit
    --work-dir WORK_DIR   the dir to save logs and models
    --amp                 enable automatic-mixed-precision training
    --resume [RESUME]     If specify checkpoint path, resume from it, while if not specify, try to auto resume from the latest checkpoint in the work directory.
    --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                            override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested
                            list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
    --launcher {none,pytorch,slurm,mpi}
                            job launcher
    --local_rank LOCAL_RANK
```

   </details>

## 📊 Evaluation [🔝](#-table-of-contents)

### 1. Single GPU

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

### 2. Multi GPU

```shell
CUDA_VISIBLE_DEVICES=x bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

You could run `python tools/test.py --help` to get detailed information of this scripts.

<details>
<summary> Detailed arguments </summary>

```
positional arguments:
    config                test config file path
    checkpoint            checkpoint file

optional arguments:
    -h, --help            show this help message and exit
    --work-dir WORK_DIR   the directory to save the file containing evaluation metrics
    --out OUT             output result file (must be a .pkl file) in pickle format
    --json-prefix JSON_PREFIX
                            the prefix of the output json file without perform evaluation, which is useful when you want to format the result to a specific format and submit it to the test server
    --tta                 Whether to use test time augmentation
    --show                show prediction results
    --deploy              Switch model to deployment mode
    --show-dir SHOW_DIR   directory where painted images will be saved. If specified, it will be automatically saved to the work_dir/timestamp/show_dir
    --wait-time WAIT_TIME
                            the interval of show (s)
    --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                            override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested
                            list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
    --launcher {none,pytorch,slurm,mpi}
                            job launcher
    --local_rank LOCAL_RANK
```

</details>

## 📦 Deployment [🔝](#-table-of-contents)

```shell
# Build docker images
docker build docker/mmdeploy/ -t mmdeploy:inside --build-arg USE_SRC_INSIDE=true
# Run docker container
docker run --gpus all --name mmdeploy_yoloms -dit mmdeploy:inside
# Convert ${CONFIG_FILE}
python tools/misc/print_config.py ${O_CONFIG_FILE} --save-path ${CONFIG_FILE}

# Copy local file into docker container
docker cp deploy.sh mmdeploy_yoloms:/root/workspace
docker cp ${DEPLOY_CONFIG_FILE}  mmdeploy_yoloms:/root/workspace/${DEPLOY_CONFIG_FILE}
docker cp ${CONFIG_FILE} mmdeploy_yoloms:/root/workspace/${CONFIG_FILE}
docker cp ${CHECKPOINT_FILE} mmdeploy_yoloms:/root/workspace/${CHECKPOINT_FILE}

# Start docker container
docker start mmdeploy_yoloms
# Attach docker container
docker attach mmdeploy_yoloms

# Run the deployment shell
sh deploy.sh ${DEPLOY_CONFIG_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SAVE_DIR}
# Copy the results to local
docker cp mmdeploy_yoloms:/root/workspace/${SAVE_DIR} ${SAVE_DIR}
```

- **DEPLOY_CONFIG_FILE**: Config file for deployment.
- **O_CONFIG_FILE**: Original config file of model.
- **CONFIG_FILE**: Converted config file of model.
- **CHECKPOINT_FILE**: Checkpoint of model.
- **SAVE_DIR**: Save dir.

### 1. Test FPS

#### 1.1 Deployed Model

```shell
# Copy local file into docker container
docker cp ${DATA_DIR} mmdeploy_yoloms:/root/workspace/${DATA_DIR}
docker cp fps.sh mmdeploy_yoloms:/root/workspace
# Start docker container
docker start mmdeploy_yoloms
# Attach docker container
docker attach mmdeploy_yoloms
# In docker container
# Run the FPS shell
python mmdeploy/tools/profiler.py ${DEPLOY_CONFIG_FILE} \
                                    ${CONFIG_FILE} \
                                    ${DATASET} \
                                    --model ${PROFILER_MODEL} \
                                    --device ${DEVICE}
```

#### 1.2 Undeployed Model

```shell
python tools/analysis_tools/benchmark.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

#### 1.3 Test FLOPs and Params

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} --shape 640 640 [optional arguments]
```