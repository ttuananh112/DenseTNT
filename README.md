# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.

## Quick Start

Requires:

* Python 3.6+
* pytorch 1.6+

### 1) Install packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse API

https://github.com/argoai/argoverse-api

### 3) Compile Cython
Compile a .pyx file into a C file using Cython:


⚠️*Recompiling is needed every time the pyx files are changed.*
``` bash
cd src/
cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

## Performance

Results on Argoverse motion forecasting validation set:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow">minADE</th>
    <th class="tg-c3ow">minFDE</th>
    <th class="tg-c3ow">Miss Rate</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization</td>
    <td class="tg-c3ow">0.80</td>
    <td class="tg-c3ow">1.27</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization (minFDE)</td>
    <td class="tg-c3ow">0.73</td>
    <td class="tg-c3ow">1.05</td>
    <td class="tg-c3ow">9.8%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ goal set predictor (online)</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
</tbody>
</table>

## Models
[Path to models](https://vingroupjsc.sharepoint.com/sites/Vantix-PAAS/Shared%20Documents/Prediction/models)

### DenseTNT

#### Train
```bash
# for carla
bash train_carla.sh
# for argoverse
bash train.sh
```

#### Evaluate
```bash
bash val.sh
# for optimize mFDE
bash val_MRmFDE.sh
```

#### Comparision
```bash
bash comparision.sh
```

#### Inference
```bash
bash inference.sh
```


## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{densetnt,
  title={Densetnt: End-to-end trajectory prediction from dense goal sets},
  author={Gu, Junru and Sun, Chen and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15303--15312},
  year={2021}
}
```
