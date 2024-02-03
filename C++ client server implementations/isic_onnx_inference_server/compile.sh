g++ -g http_server_onnx_classifier.cpp onnx_isic_inferencer.cpp base64.cpp -lonnxruntime -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread -lboost_system -o server_sample
g++ -g test_client_base64.cpp onnx_isic_inferencer.cpp base64.cpp -lonnxruntime -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread -lboost_system -o client_sample


                                                                                                                                                                                                                    