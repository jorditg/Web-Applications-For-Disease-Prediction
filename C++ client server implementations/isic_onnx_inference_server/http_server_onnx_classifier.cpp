#include "client_http.hpp"
#include "server_http.hpp"
#include <future>

// Added for the json
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "onnx_isic_inferencer.h"

#include "object_pool.hpp"

using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;
using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;
using I = OnnxIsicInferencer;

int main() {

  // Define a pool of inferencers
  // Allocate ONNX inference objects
  std::size_t init_sz = 5;
  std::size_t max_sz = 10;
  std::string model_filename = "model0.onnx";

  ObjectPool<I, std::string &> pool(init_sz, max_sz, model_filename);
  std::cout << "Initialized object pool with max size of " << pool.max_size() << std::endl;
  std::cout << "Size of the pool: " << pool.size() << std::endl;
  std::cout << "Available objects: " << pool.free() << std::endl;

  // HTTP-server at port 8080 using 1 thread
  // Unless you do more heavy non-threaded processing in the resources,
  // 1 thread is usually faster than several threads
  HttpServer server;
  server.config.port = 8080;

  // POST-example for the path /json, responds firstName+" "+lastName from the posted json
  // Responds with an appropriate error message if the posted json is not valid, or if firstName or lastName is missing
  // Example posted json:
  // {
  //   "firstName": "John",
  //   "lastName": "Smith",
  //   "age": 25
  // }
  server.resource["^/json$"]["POST"] = [&pool](std::shared_ptr<HttpServer::Response> response, std::shared_ptr<HttpServer::Request> request) {
    try {
      boost::property_tree::ptree pt;
      boost::property_tree::read_json(request->content, pt);

      std::string image_name = pt.get<std::string>("imageName");
      std::string image_base64 = pt.get<std::string>("imageBase64");
      //std::cout << image_base64;
      auto inf = pool.get();
      if(inf) {
        std::string result = inf->PredictJPEGBase64(image_base64);
        response->write(result + "\r\n\r\n");
        // object automatically released by object_pool defined delete function
      } else {
        response->write("Server busy.\r\n\r\nRetry later.\r\n\r\n");       
      }
      //std::string name = image_name + image_base64;
      //*response << "HTTP/1.1 200 OK\r\n"
      //          << "Content-Length: " << name.length() << "\r\n\r\n"
      //          << name;
    }
    catch(const std::exception &e) {
      *response << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << strlen(e.what()) << "\r\n\r\n"
                << e.what();
    }


    // Alternatively, using a convenience function:
    // try {
    //     ptree pt;
    //     read_json(request->content, pt);

    //     auto name=pt.get<string>("firstName")+" "+pt.get<string>("lastName");
    //     response->write(name);
    // }
    // catch(const std::exception &e) {
    //     response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    // }
  };

  server.on_error = [](std::shared_ptr<HttpServer::Request> /*request*/, const SimpleWeb::error_code & /*ec*/) {
    // Handle errors here
    // Note that connection timeouts will also call this handle with ec set to SimpleWeb::errc::operation_canceled
  };

  // Start server and receive assigned port when server is listening for requests
  std::promise<unsigned short> server_port;
  std::thread server_thread([&server, &server_port]() {
    // Start server
    server.start([&server_port](unsigned short port) {
      server_port.set_value(port);
    });
  });
  std::cout << "Server listening on port " << server_port.get_future().get() << std::endl << std::endl;

  // Client sample
  std::string filename = "sample.base64";
  std::string str(std::istreambuf_iterator<char>(std::ifstream(filename).rdbuf()), std::istreambuf_iterator<char>());

  boost::property_tree::ptree pt;
  pt.put("imageName", "ISIC_0034321.jpg");
  pt.put("imageBase64", str);

  std::stringstream ss;
  boost::property_tree::json_parser::write_json(ss, pt);
  std::string json_string = ss.str();

  // Asynchronous request sample
  {
    HttpClient client("localhost:8080");
    std::cout << "Example POST request to http://localhost:8080/json" << std::endl;
    client.request("POST", "/json", json_string, [](std::shared_ptr<HttpClient::Response> response, const SimpleWeb::error_code &ec) {
      if(!ec)
        std::cout << "Response content: " << response->content.rdbuf() << std::endl;
    });
    client.io_service->run();
  }

  server_thread.join();
}

