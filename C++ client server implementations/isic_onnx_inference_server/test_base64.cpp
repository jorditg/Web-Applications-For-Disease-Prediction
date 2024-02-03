
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "base64.h"


int main()
{
  std::string filename = "sample.base64";
  std::string base64image(std::istreambuf_iterator<char>(std::ifstream(filename).rdbuf()), std::istreambuf_iterator<char>());
  std::remove_if(base64image.begin(), base64image.end(), isspace);
  std::cout << base64image << std::endl;
  std::string str = base64_decode(base64image);
  std::vector<char> decoded(str.begin(), str.end());
  std::cout << str << std::endl;
  cv::Mat img = cv::imdecode(decoded, cv::IMREAD_COLOR);
  //cv::Mat img = cv::imdecode(cv::Mat(decoded), CV_LOAD_IMAGE_UNCHANGED);
  std::cout << base64image.substr(0,10) 
            << " COLS: " << img.cols 
            << "ROWS: " << img.rows 
            << std::endl;

}