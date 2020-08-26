#pragma once

class ObjectDetector {
 private:
#if 0

  samplesCommon::OnnxSampleParams mParams;  //!< The parameters for the sample.

  nvinfer1::Dims mInputDims;   //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims;  //!< The dimensions of the output to the network.
  int mNumber{0};              //!< The number to classify

  std::shared_ptr<nvinfer1::ICudaEngine>
      mEngine;  //!< The TensorRT engine used to run the network

  //!
  //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
  //!
  bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                        SampleUniquePtr<nvonnxparser::IParser> &parser);

  //!
  //! \brief Reads the input  and stores the result in a managed buffer
  //!
  bool processInput(const samplesCommon::BufferManager &buffers);

  //!
  //! \brief Classifies digits and verify result
  //!
  bool verifyOutput(const samplesCommon::BufferManager &buffers);
#endif
 public:
  ObjectDetector();
  ~ObjectDetector();
  bool Build();
  bool Infer();
};
