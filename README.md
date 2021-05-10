ONNX and Neural Network Modeling Framework Format Testing Script
=======
This code produces results published in

P. Plagwitz, F. Hannig, M. Ströbel, C. Strohmeyer and J. Teich, „*A safari through FPGA-based neural network compilation and design automation flows*“, in Proceedings of the 29th IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), IEEE, 2021

It can be cited as
```
@inproceedings{safari,
  author={Plagwitz, Patrick and Hannig, Frank and Ströbel, Martin and Strohmeyer, Christoph and Teich, Jürgen},
  booktitle={Proceedings of the 29th IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM)}, 
  title={A Safari through {FPGA}-based Neural Network Compilation and Design Automation Flows}, 
  year={2021},
  organization={IEEE}}
  ```

Usage
=========
 1. Clone the repo.
 1. Run 
 ``` $ bash download-onnx-model-zoo.sh ```, which downloads the ONNX Model Zoo repo (requires 2GB-3GB disk space) and the Keras ShuffleNet model.
 1. Use the testNetwork function in the script ```onnxruntest.py``` with any combination of network and format.

Dependencies
==========
Following software versions were tested and need to be installed:

| Software | Version |
| ---- | ---- |
| Keras | 2.4.3 | 
| onnx2keras | 881dcbd9 (commit) | 
| keras2onnx | 1.7.0 |
| PyTorch | 1.6.0 |
| TensorFlow | 2.2.0 |
| onnx-rf | 1.7.0 |
| tf2onnx | 1c9c02d220 (commit)
| MXNet | 1.7.0.post1
