# instalar previamente includes y libs en /usr/local y ejecutar sudo ldconfig
g++ -g main.cpp onnx_isic_inferencer.cpp base64.cpp -lonnxruntime -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
