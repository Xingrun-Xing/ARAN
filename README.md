# ARAN

This is the implementation of our ARAN. We provide the architectures and pretrained models as above.<br>
This readme is updating now, and if you need more detailed information, please contact me at 15137162936@163.com.<br> 

## Run(the same as [WDSR](https://github.com/JiahuiYu/wdsr_ntire2018))
### Requirements: 
* Install [PyTorch](https://pytorch.org/) (tested on release 0.4.0 and 0.4.1).
* Clone [EDSR-Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch/tree/95f0571aa74ddf9dd01ff093081916d6f17d53f9) as backbone training framework.
### Training and Validation:
* Copy [aran_cx_sx.py](/models), into `EDSR-PyTorch/src/model/`.
* Modify `EDSR-PyTorch/src/option.py` and `EDSR-PyTorch/src/demo.sh` to support `--[r,g,b]_mean` option (please find reference in issue [#7](https://github.com/JiahuiYu/wdsr_ntire2018/issues/7) in the [WDSR](https://github.com/JiahuiYu/wdsr_ntire2018) issues).
* Launch training with [EDSR-Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch/tree/95f0571aa74ddf9dd01ff093081916d6f17d53f9) as backbone training framework.
