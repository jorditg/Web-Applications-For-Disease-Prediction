#include <string>
#include <iostream>
#include "onnx_isic_inferencer.h"

int main(int argc, char *argv[]) {
  const char * model = "model0.onnx";
  OnnxIsicInferencer inf = OnnxIsicInferencer(model);

  const char * filename = argv[1];

  //inf.PredictSample();

  //inf.PredictJPEGFile(filename);

   // Base64 encoded file predictor
  std::string str(std::istreambuf_iterator<char>(std::ifstream(filename).rdbuf()), std::istreambuf_iterator<char>());
  inf.PredictJPEGBase64(str);
}
