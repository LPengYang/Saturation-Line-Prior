# Single Image Dehazing Using Saturation Line Prior (2023 TIP) 

<hr />

> **Abstract:** *—Saturation information in hazy images is conducive
to effective haze removal, However, existing saturation-based
dehazing methods just focus on the saturation value of each
pixel itself, while the higher-level distribution characteristic
between pixels regarding saturation remains to be harnessed.
In this paper, we observe that the pixels, which share the same
surface reflectance coefficient in the local patches of haze-free
images, exhibit a linear relationship between their saturation
component and the reciprocal of their brightness component in
the corresponding hazy images normalized by atmospheric light.
Furthermore, the intercept of the line described by this linear
relationship on the saturation axis is exactly the saturation value
of these pixels in the haze-free images. Using this characteristic of
saturation, termed saturation line prior (SLP), the transmission
estimation is translated into the construction of saturation lines.
Accordingly, a new dehazing framework using SLP is proposed,
which employs the intrinsic relevance between pixels to achieve a
reliable saturation line construction for transmission estimation.
This approach can recover the fine details and attain realistic
colors from hazy scenes, resulting in a remarkable visibility
improvement. Extensive experiments in real-world and synthetic
hazy images show that the proposed method performs favorably
against state-of-the-art dehazing methods.* 
<hr />

## Demo

We have provided two version of demos, i,e., the concise verison (demo_concise.mlx) that only returns dehazing results for input hazy images, and the detailed verison (demo_detailed.mlx) that outputs all intermidiate results with visualization. 

## Demonstration of Saturation Line Prior (SLP)

![image](https://github.com/LPengYang/Saturation_Line_Prior/blob/main/demonstration%20figures/demonstrastion_slp_process.png) 

![image](https://github.com/LPengYang/Saturation_Line_Prior/blob/main/demonstration%20figures/Concept.png) 

![image](https://github.com/LPengYang/Saturation_Line_Prior/blob/main/demonstration%20figures/more_examples.png) 


## Citation
If you use our work, please consider citing:

    @ARTICLE{10141557,
    author={Ling, Pengyang and Chen, Huaian and Tan, Xiao and Jin, Yi and Chen, Enhong},
    journal={IEEE Transactions on Image Processing}, 
    title={Single Image Dehazing Using Saturation Line Prior}, 
    year={2023},
    volume={32},
    number={},
    pages={3238-3253},
    doi={10.1109/TIP.2023.3279980}}
