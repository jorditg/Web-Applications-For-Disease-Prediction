#include "client_http.hpp"
#include <future>

// Added for the json
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

int main() {

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

}

