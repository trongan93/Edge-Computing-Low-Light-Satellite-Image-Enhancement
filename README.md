# Edge-Computing-Low-Light-Satellite-Image-Enhancement
This repository contains the full source code of the paper:

Trong-An Bui, Pei-Jun Lee, Chun-Sheng Liang, Pei-Hsiang Hsu, Shiuan-Hal Shiu and Chen-Kai Tsai, "**Edge-Computing-Enabled Deep Learning Approach for Low-Light Satellite Image Enhancement**," 
in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 4071-4083, 2024, doi: 10.1109/JSTARS.2024.3357093.

**Abstract**: Edge computing enables rapid data processing and decision-making on satellite payloads. Deploying deep learning-based techniques for low-light image enhancement improves early detection and tracking accuracy on satellite platforms, but it faces challenges due to limited computational resources. This article proposes an edge-computing-enabled inference model specifically designed onboard satellites. The proposed model follows an encoder–decoder architecture to generate the illumination map with low multiplication matrix complexity, 25.52 GMac of $1920 \times 1200$ image size. To reduce nanosatellite hardware consumption with a single-precision floating-point format, the edge-computing-enabled inference model proposes a quantized convolution that computes signed values. The proposed inference model is deployed on Arm Cortex-M3 microcontrollers onboard satellite payload (86.74 times faster than normal convolution model) but also has a similar quality with the low-light enhanced in full-precision computing of lightweight training model by using the peak signal-to-noise ratio (average of 28.94) and structural similarity index (average of 0.85) metrics.

keywords: {Satellites;Computational modeling;Image edge detection;Image enhancement;Edge computing;Payloads;Deep learning;Deep learning;edge computing;edge-computing-enabled;image enhancement;low light satellite images;onboard},
URL: https://ieeexplore.ieee.org/abstract/document/10412123


**Step 1**: Training **Proposed Lightweight Model** 
[lowlight_train.py](/Low-Light-Satellite-Image-Enhancement-Lightweight-Training-Model/lowlight_train.py)

**Step 2**: Compress the training weight **Post-training quantization**
[compress_trained_weights.py](/Low-Light-Satellite-Image-Enhancement-Lightweight-Training-Model/compress_trained_weights.py)

**Step 3**: Inference with the **Proposed Model**, including **Quantized Convolution** and **Piece-Wise** sigmoid function 
[main.cpp](/Inference/main.cpp)

**Experiment Results**

**Test case 1**
![Result 1 Edge-Computing-Low-Light-Satellite-Image-Enhancement](/Inference/paper_case1/_mnt_d_Seaport_satellite_images_lng_3.110000_lat_50.560000_sentinel_2_rgb_enhanced_onboard.png)


**Test case 2**
![Result 2 Edge-Computing-Low-Light-Satellite-Image-Enhancement](/Inference/paper_case2/_mnt_d_Seaport_satellite_images_lng_4.240000_lat_51.520000_sentinel_2_rgb_enhanced_onboard.png)

**Test case 3**
![Result 3 Edge-Computing-Low-Light-Satellite-Image-Enhancement](/Inference/paper_case3/_mnt_d_Seaport_satellite_images_lng_5.270000_lat_52.590000_landsat_8_rgb_enhanced_onboard.png)

Please reference this paper in your manuscript when you use this source code.
