# TensorRT for keras
## Step to run
 - Convert your model files (h5) into a frozen graph using the freeze graph code
 - Run the optimize script to get a new graph with tensorrt operations 
 
## Steps to setup Tensorflow with TensorRT support (Ubuntu 16.04 LTS)
- Install necessary cuda and cudnn libararies. (I got success with cuda-9.0 and cudnn-7.0)
- Install TensorRT from official repos (Building from source works too)
- Build tensorflow from source and enable TensorRT support in configuration script.

## Results
- Results available for a denesnet model trained on a custom dataset
- Inference not done in batch mode
- Time in seconds
![Alt Text](https://github.com/srihari-humbarwadi/TensorRT-for-keras/blob/master/results_densenet.jpeg)
