#ifndef ONNXISICINFERENCER_H
#define ONNXISICINFERENCER_H

#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp"

class OnnxIsicInferencer
{
private:
    // inference required sizes
    const int64_t PRED_SIZE = 224;
    const int64_t PRED_CHANNELS = 3;
    const int64_t input_tensor_size = PRED_SIZE * PRED_SIZE * PRED_CHANNELS;

    // MEAN AND STDEV used by our model for normalize images previous to inference
    const float RGB_mean[3] =  {166.43850410293402, 133.5872994553671, 132.33856917079888};
    const float RGB_stdev[3] = {59.679343313897874, 53.83690126788451, 56.618447349633676};

    // model characteristics
    const std::vector<int64_t> input_node_dims = {1, PRED_CHANNELS, PRED_SIZE, PRED_SIZE};
    const std::vector<const char*> input_node_names = {"input"};
    const std::vector<const char*> output_node_names = {"output"};

    std::string model_path;
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::SessionOptions> session_options;
    std::shared_ptr<Ort::Session> session;
    
    // auxiliary functions
    void mat2vector(cv::Mat &mat, std::vector<float> &array);
public:
    OnnxIsicInferencer(std::string & model_path);
    int PredictSample();
    std::string PredictJPEGBase64(std::string & base64image);
    std::string PredictJPEGFile(const char * filename);
    std::string PredictJPEG(cv::Mat & img);
};

#endif // ONNXISICINFERENCER_H
