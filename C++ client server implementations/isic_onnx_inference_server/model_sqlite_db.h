#include <sqlite3.h>
#include <iostream>

class model_sqlite_db
{
public:
  model_sqlite_db();

  bool is_user_registered(std::string & id_user);
  bool is_image_classified(std::string & md5file);

  std::string retrieve_image_classification(std::string &md5file);

  bool register_user(std::string &id_user);
  bool register_image_classification(std::string &id_user, std::string &md5file, std::string &classification);

private:
	sqlite3 *db;

}