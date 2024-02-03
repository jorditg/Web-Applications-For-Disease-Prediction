#include "onnx_isic_inferencer.h"

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iterator>

#include <opencv2/opencv.hpp>

#include "base64.h"
OnnxIsicInferencer::OnnxIsicInferencer(const char * model_path)
{
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    // initialize session options if needed
    session_options.reset(new Ort::SessionOptions());
    session_options->SetIntraOpNumThreads(1);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
    // #include "cuda_provider_factory.h"
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // set model path
    this->model_path = model_path;

    printf("Using Onnxruntime C++ API\n");
    session.reset(new Ort::Session(*env.get(), model_path, *session_options.get()));
}

int OnnxIsicInferencer::PredictSample()
{
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;
  // print number of model input nodes
  size_t num_input_nodes = session->GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session->GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    input_node_dims[0] = 1; // dimension in model set to -1. Reset to actual value
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"output"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);


  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());


  // score model & input tensor, get back output tensor
  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  // score the model, and print scores for first 9 classes
  for (int i = 0; i < 9; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  printf("Done!\n");
  return 0;
}

// copy mat to vector and normalize
void OnnxIsicInferencer::mat2vector(cv::Mat &mat, std::vector<float> &array)
{
  // mat indexes are (HEIGHT, WIDTH, CHANNEL) and array indexes (CHANNEL, HEIGHT, WIDTH)
  // we take this into account in the loop
  for (int c = 0; c < PRED_CHANNELS; c++) { // channels
    for (int i = 0; i < PRED_SIZE; i++) { // rows
      for (int j = 0; j < PRED_SIZE; j++) { // cols
        size_t idx_fr = PRED_CHANNELS*(PRED_SIZE*i + j) + c;
        size_t idx_to = PRED_SIZE*PRED_SIZE*c + PRED_SIZE*i + j;
        array[idx_to] = (((float) mat.data[idx_fr]) - RGB_mean[c])/RGB_stdev[c];
      }
    }
  }
}

int OnnxIsicInferencer::PredictJPEGBase64(std::string & base64image) {
   std::string str = base64_decode(base64image);
   std::vector<uint8_t> decoded(str.begin(), str.end());
   cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
   PredictJPEG(img);
}

int OnnxIsicInferencer::PredictJPEGFile(const char * filename) {
   cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
   PredictJPEG(img);
}

int OnnxIsicInferencer::PredictJPEG(cv::Mat & img_orig)
{
  /*
     Comprobada la inferencia de script de python vs C++, para un tamaño de imagen 224x224 igual, los resultados
     difieren un poco, en los decimales de los resultados.

     Para imágenes de mayor tamaño los resultados difieren más, seguramente debido a diferencias en los algoritmos
     de interpolación añadidos a los indicados en el párrafo anterior.

     Comprobado que realizando inferencia de exactamente la misma matriz, los resultados coinciden plenamente.
   */

  // resize to prediction size
  cv::Mat img_resized;
  cv::resize(img_orig, img_resized, cv::Size(PRED_SIZE,PRED_SIZE), 0, 0, cv::INTER_LINEAR);

  // changes BGR OpenCV standard colorspace to RGB (required by the model)
  cv::Mat img;
  cv::cvtColor(img_resized, img, cv::COLOR_BGR2RGB); // after this, img dimensions: (224,224,3) where 3 are RGB channels
   
  //printf("Print 10 values of RGB channel\n");
  //for(int i=0;i<30;i++) {
  //  printf("%i ", img.data[i]);
  //  if((i+1)%3 == 0) printf("\n");
  //}
  //printf("\n");

  // assert correct image size (pending include channels in assertion)
  assert(img.rows == PRED_SIZE && img.cols == PRED_SIZE);

  std::vector<float> input_tensor_values(input_tensor_size);

  // copy array from image to vector and normalize
  mat2vector(img, input_tensor_values);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  // score the model, and print scores for first 9 classes
  for (int i = 0; i < 9; i++)
    printf("%f ", floatarr[i]);

  printf("\nDone!\n");
  return 0;  
}


