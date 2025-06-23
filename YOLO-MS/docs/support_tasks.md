## 🏗️ Supported Tasks [🔙](../README.md)

### Object Detection [🔝](#top)

Please refer to [Model Zoos](model_zoos.md).

### Instance Segmentation [🔝](#top)

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$bbox AP$</th>
      <th>$seg AP$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>RTMDet-T</td>
      <td>300</td>
      <td>5.6M</td>
      <td>11.8G</td>
      <td>40.5</td>
      <td>35.4</td>
      <td>–</td>
      <td>–</td>
    </tr>
    <tr>
      <td>YOLO-MS-XS</td>
      <td>300</td>
      <td>5.1M</td>
      <td>12.9G</td>
      <td>42.3</td>
      <td>20.9</td>
      <td><a href="../mmyolo/configs/yoloms/ins/yoloms-xs-ins_syncbn_fast_8xb16-300e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1lV5dBF3jBuJWEkQgBDQScBWhQ42oSJdX/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/other_tasks/yoloms-xs-ins_syncbn_fast_8xb16-300e_coco.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### Rotated Object Detection  [🔝](#top)

The dataset is [DOTA](https://captain-whu.github.io/DOTA/).

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$AP50$</th>
      <th>$AP75$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>YOLO-MS-R-XS</td>
      <td>36</td>
      <td>4.4M</td>
      <td>22G</td>
      <td>61.8</td>
      <td>88</td>
      <td>70.3</td>
      <td><a href="../mmyolo/configs/yoloms/dota/yoloms-xs_syncbn_fast_2xb4-36e_dota.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1j3F2Y7PPgD3xOZUxRzyxLoIHWnnkaiLr/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/dota/yoloms-xs_syncbn_fast_2xb4-36e_dota.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS-R-S</td>
      <td>36</td>
      <td>7.4M</td>
      <td>38G</td>
      <td>63.8</td>
      <td>88.7</td>
      <td>73.6</td>
      <td><a href="../mmyolo/configs/yoloms/dota/yoloms-s_syncbn_fast_2xb4-36e_dota.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1xFMmPDYyMGtwEyf8sbEFj2bOivGBGk7t/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/dota/yoloms-s_syncbn_fast_2xb4-36e_dota.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS-R</td>
      <td>36</td>
      <td>20.0M</td>
      <td>99G</td>
      <td>66.9</td>
      <td>89.9</td>
      <td>77.8</td>
      <td><a href="../mmyolo/configs/yoloms/dota/yoloms_syncbn_fast_2xb4-36e_dota.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1wpJ7WsK2izRsNu_bMvHGYFjscmA-ef0a/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/dota/yoloms_syncbn_fast_2xb4-36e_dota.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS-R-L</td>
      <td>36</td>
      <td>42.7M</td>
      <td>206G</td>
      <td>68.6</td>
      <td>90.6</td>
      <td>80.7</td>
      <td><a href="../mmyolo/configs/yoloms/dota/yoloms-l_syncbn_fast_2xb4-36e_dota.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1F2-k9DALydJhXS1Um_EiweixnwbQzZOI/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/dota/yoloms-l_syncbn_fast_2xb4-36e_dota.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### Detection for Crowded Scene  [🔝](#top)

The dataset is [Crowdhuman](https://www.crowdhuman.org/).

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$mMR$</th>
      <th>$JI$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>RTMDet-T</td>
      <td>300</td>
      <td>4.9M</td>
      <td>8.0G</td>
      <td>85.8</td>
      <td>47.2</td>
      <td>76</td>
      <td>–</td>
      <td>–</td>
    </tr>
    <tr>
      <td>YOLO-MS-XS</td>
      <td>300</td>
      <td>5.1M</td>
      <td>8.6G</td>
      <td>87</td>
      <td>46.3</td>
      <td>78.1</td>
      <td><a href="../mmyolo/configs/yoloms/crowdhuman/yoloms-xs_syncbn_fast_8xb32-300e_crowdhuman.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1fePjFqNCE-01wvYFCFVt4JYepOs-zl2C/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/other_tasks/yoloms-xs_syncbn_fast_8xb32-300e_crowdhuman.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### Detection for Underwater Scene  [🔝](#top)

The dataset is [RUOD](https://github.com/dlut-dimt/RUOD).

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>RTMDet-T</td>
      <td>300</td>
      <td>4.9M</td>
      <td>8.0G</td>
      <td>63.6</td>
      <td><a href="../mmyolo/configs/yoloms/ruod/rtmdet-tiny_syncbn_fast_8xb8-300e_ruod.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1NXMyv-0kBNys4A0tsl5WL6ilpIyeCOmS/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/other_tasks/rtmdet-tiny_syncbn_fast_8xb8-300e_ruod.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS-XS</td>
      <td>300</td>
      <td>5.1M</td>
      <td>8.6G</td>
      <td>63.9</td>
      <td><a href="../mmyolo/configs/yoloms/ruod/yoloms-xs_syncbn_fast_8xb8-300e_ruod.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/15DzZT6gXXQszTMzHleu80e7lnuEhCWm3/view?usp=drive_link">model</a> |
        <a href="logs/yoloms/other_tasks/yoloms-xs_syncbn_fast_8xb8-300e_ruod.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### Detection for Foggy Scene  [🔝](#top)

The dataset is [RTSS](https://arxiv.org/abs/1712.04143).

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>RTMDet-T</td>
      <td>100</td>
      <td>4.9M</td>
      <td>8.0G</td>
      <td>59.2</td>
      <td><a href="../mmyolo/configs/yoloms/rtss/rtmdet-tiny_syncbn_fast_8xb8-100e_rtts.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1_CY-4ONm5JbC9_d6eBvplFY_etTsp0G7/view?usp=drive_link">model</a> |
          <a href="logs/yoloms/other_tasks/rtmdet-tiny_syncbn_fast_8xb8-100e_rtts.json">Log</a></td>
    </tr>
    <tr>
      <td>YOLO-MS-XS</td>
      <td>100</td>
      <td>5.1M</td>
      <td>8.6G</td>
      <td>59.7</td>
      <td><a href="../mmyolo/configs/yoloms/rtss/yoloms-xs_syncbn_fast_8xb8-100e_rtts.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/19oSpW57zUmm55ZHKpa543PiK5m1Y8jjp/view?usp=drive_link">model</a> |
          <a href="logs/yoloms/other_tasks/yoloms-xs_syncbn_fast_8xb8-100e_rtts.json">Log</a></td>
    </tr>
  </tbody>
</table>

### Detection for RAW image in diverse conditions  [🔝](#top)

Please refer to [AODRaw](https://github.com/lzyhha/AODRaw)
