import numpy as np 
import onnxruntime
import onnx
from onnx import numpy_helper
import json
import time
import os

from onnx2keras import onnx_to_keras
import onnxmltools
import tensorflow as tf
import torchvision.models as models
import torch

def getOnnxInputNames(fileName):
  session = onnxruntime.InferenceSession(fileName)
  return [i.name for i in session.get_inputs()]
def getOnnxOutputNames(fileName):
  session = onnxruntime.InferenceSession(fileName)
  return [o.name for o in session.get_outputs()]
def getOnnxInputShapes(fileName):
  session = onnxruntime.InferenceSession(fileName)
  return [i.shape for i in session.get_inputs()]

def convertToKerasAndBack(fileName, outputFileName, inputs):
  ret = None
  onnxModel = onnx.load(fileName)
  kwArgs = dict()
  if "shuffle" in fileName:
    kwArgs["input_shapes"] = [1, 3, 224, 224]
    import shufflenet
    import keras
    kerasModel = shufflenet.ShuffleNet(groups=3)
    kerasModel.load_weights("keras-shufflenet/weights/ShuffleNet_1X_g3_br_0.25_373.hdf5")
    kerasModel.compile(
              optimizer=keras.optimizers.SGD(lr=.05, decay=5e-4, momentum=0.9),
              metrics=['accuracy'],
              loss='categorical_crossentropy')
    ret = kerasModel.predict(inputs[0][0].transpose(0, 2, 3, 1))
  else:
    kerasModel = onnx_to_keras(onnxModel, getOnnxInputNames(fileName),
        verbose=False, **kwArgs)

  #tf.keras.utils.plot_model(kerasModel, show_shapes=True)
  backconvOnnxModel = onnxmltools.convert_keras(kerasModel)
  onnxmltools.utils.save_model(backconvOnnxModel, outputFileName)
  return ret

class VoltageNet(torch.nn.Module):
  def __init__(self):
    super(VoltageNet, self).__init__()
    self.layer1 = torch.nn.Conv2d(1, 16, 16, 16)
    self.tanh = torch.nn.Tanh()
    self.flatten = torch.nn.Flatten()
    self.layer2 = torch.nn.Linear(16, 4)
    self.output = torch.nn.Softmax()

  def forward(self, x):
    x = self.layer1(x)
    x = self.tanh(x)
    x = self.flatten(x)
    x = self.layer2(x)
    x = self.output(x)
    return x

def evaluatePyTorchModel(model, inputs):
  #print(inputs[0][0].shape)
  ret = []
  for i in inputs:
    torchInputs = [torch.from_numpy(vector) for vector in i]
    result = model(*torchInputs)
    ret.append([result.detach().numpy()])
  return ret

def convertToPyTorchAndBack(fileName, outputFileName, inputs):
  if "resnet50" in fileName:
    model = models.resnet50(pretrained=True)
  if "shufflenet" in fileName:
    model = models.shufflenet_v2_x1_0(pretrained=True)
  if "voltage" in fileName:
    model = VoltageNet()
    model.load_state_dict(torch.load("net-state"))
  model.eval()
  
  ret = evaluatePyTorchModel(model, inputs)
  
  torch.onnx.export(model,
      #torch.randn(1, 3, 224, 224),
      torch.randn(*getOnnxInputShapes(fileName)[0]),
      outputFileName,
      export_params=True,
      opset_version=10,
      do_constant_folding=True)
#      input_names=['input'],
#      output_names=['output'])
  return ret

def toFrozenGraph(concreteFunc):
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
  import numpy as np

  tf.compat.v1.disable_eager_execution()
  frozen_func = convert_variables_to_constants_v2(concreteFunc)
  return frozen_func.graph.as_graph_def()


def convertToTfAndBack(fileName, outputFileName):
  import onnx_tf
  onnxModel = onnx.load(fileName)
  tfModel = onnx_tf.backend.prepare(onnxModel)
  tfModel.export_graph("/tmp/tf-saved-model")

  import subprocess
  subprocess.check_call(["python3", "-m", "tf2onnx.convert", "--saved-model",
    "/tmp/tf-saved-model/", "--output", outputFileName])

def convertToMxNetAndBack(fileName, outputFileName, inputs):
  from mxnet.gluon.model_zoo import vision
  import mxnet.contrib.onnx as onnx_mxnet
  import mxnet
  ret = None
  if "resnet" in fileName:
    model = vision.resnet50_v2(pretrained=True)
    model.hybridize()
    ret = [model(mxnet.nd.array(inputs[0][0]))]
    model.export("mxnet-model")
    sym = "./mxnet-model-symbol.json"
    arg = "./mxnet-model-0000.params"
  else:
    sym, arg, aux = onnx_mxnet.import_model(fileName)
  onnx_mxnet.export_model(sym, arg, getOnnxInputShapes(fileName), np.float32, outputFileName)
  return ret

def loadTensor(fileName):
  if fileName.endswith("npz"):
    return np.load(fileName, encoding="bytes")["arr_0"]
    #return list(np.load(fileName, encoding="bytes")["inputs"])
  tensor = onnx.TensorProto()
  with open(fileName, "rb") as binaryFile:
    tensor.ParseFromString(binaryFile.read())
    return numpy_helper.to_array(tensor)

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocessImageNetImage(inputData):
  imgData = inputData.astype("float32")
  means = np.array([0.485, 0.456, 0.406])
  stddevs = np.array([0.229, 0.224, 0.225])
  normImgData = np.zeros(imgData.shape).astype("float32")
  for i in range(imgData.shape[0]):
    normImgData[i,:,:] = (imgData[i,:,:] / 255 - means[i]) / stddevs[i]

  return normImgData.reshape(1, 3, 224, 224).astype("float32")

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def loadTensors(pathFormat, noOfSets, noOfItems):
  ret = []
  for i in range(noOfSets):
    vector = []
    for j in range(noOfItems):
      vector.append(loadTensor(pathFormat.format(i, j)))
    ret.append(vector)
  return ret

import collections
Network = collections.namedtuple("Network", [
  "path", "dataFormat", "noOfTestData", "noOfInputs", "noOfOutputs"])



folder = "onnx-zoo/models"
networks = [
    Network("../voltage", "npz", 1, 1, 1),
    Network("vision/classification/resnet/model/resnet50-v2-7", "pb", 3, 1, 1),
    Network("vision/classification/shufflenet/model/shufflenet-v2-10", "pb",
      1, 1, 1),
    Network(
      "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11",
      "pb", 1, 2, 3),
    Network(
      "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9",
      "pb", 16, 4, 2),
    Network("vision/classification/resnet/model/resnet50-v1-7", "pb", 3, 1, 1)]

def testNetwork(framework, network):
  path = os.path.join("onnx-zoo", network.path)
  testDataDir = os.path.join(path, "test_data_set_")

  onnxPath = path + ".onnx"
  newOnnxPath = path + ".back.onnx"
  newOutputs = None

  inputs = loadTensors(testDataDir + "{0}/input_{1}." + network.dataFormat,
      network.noOfTestData, network.noOfInputs)
  refOutputs = loadTensors(testDataDir + "{0}/output_{1}." + network.dataFormat,
      network.noOfTestData, network.noOfOutputs)

  if framework == "keras":
    newOutputs = convertToKerasAndBack(onnxPath, newOnnxPath, inputs)
  if framework == "pytorch":
    newOutputs = convertToPyTorchAndBack(onnxPath, newOnnxPath, inputs)
  if framework == "mxnet":
    newOutputs = convertToMxNetAndBack(onnxPath, newOnnxPath, inputs)
  if framework == "tf":
    convertToTfAndBack(onnxPath, newOnnxPath)

  onnxPath = newOnnxPath
  if newOutputs is not None:
    refOutputs = newOutputs

  session = onnxruntime.InferenceSession(onnxPath)
  inputNames = [session.get_inputs()[i].name for i in range(network.noOfInputs)]
  print("InputNames:", inputNames)
  indexMap = {0: 0, 1: 1, 2: 2, 3: 3}
  if "bidaf" in path:
    indexMap = {0: 0, 1: 2, 2: 1, 3: 3}
  if "shuffle" in path and framework == "keras":
    print(inputs[0][0].shape)
    inputs[0] = [inputs[0][0].transpose(0, 2, 3, 1)]

  outputs = [session.run([], dict(
    (inputNames[j], inputs[i][indexMap[j]]) for j in range(network.noOfInputs)))
    for i in range(network.noOfTestData)]
  print(outputs)

  for refO, o in zip(refOutputs, outputs):
    for i in range(network.noOfOutputs):
      np.testing.assert_almost_equal(refO[i], o[i], 3)
  print("Successful")

testNetwork("tf", networks[3])
