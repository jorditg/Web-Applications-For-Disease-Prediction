// A small example program using SQLite with C++
#include <iostream>
#include <sqlite3.h>


static int db_build(std::string &database_filename)
{
   const char sql1[] = "CREATE TABLE CLASSIFICATIONS("  
         "ID_IMAGE           CHARACTER(32)  PRIMARY KEY NOT NULL, "  // MD5 FILE
		 "IMAGE_EXTENSION    VARCHAR(5)                 NOT NULL, "  // FILE FORMAT EXTENSION  
         "CLASSIFICATION     TEXT                       NOT NULL);"; // CLASS RESULT IN TEXT FORMAT		  

   const char sql2[] = "CREATE TABLE USERS(ID_USER CHARACTER(32) PRIMARY KEY NOT NULL);";
   // 128-bit ON INSTALLATION TIME RANDOM GENERATED USER IDENTIFIER

   const char sql3[] = "CREATE TABLE USER_IMAGE_PAIRS(";
                       "ID_USER  CHARACTER(32) NOT NULL,"
                       "ID_IMAGE CHARACTER(32) NOT NULL,"
					   "PRIMARY KEY (ID_USER, ID_IMAGE_MD5));";

   sqlite3 *db;   
   sqlite3_open(database_filename.c_str(), &db);
   sqlite3_exec(db, sql1, NULL, NULL, NULL);
   sqlite3_exec(db, sql2, NULL, NULL, NULL);
   sqlite3_exec(db, sql3, NULL, NULL, NULL);
   sqlite3_close(db);

   return 0;
}

static int callback(void *NotUsed, int argc, char **argv, char **azColName)
{
	int i;
	for(i=0; i<argc; i++)
	{
		cout<<azColName[i]<<" = " << (argv[i] ? argv[i] : "NULL")<<"\n";
	}
	cout<<"\n";
	return 0;
}
  
int main()
{
	const int STATEMENTS = 8;
	sqlite3 *db;
	char *zErrMsg = 0;
	const char *pSQL[STATEMENTS];
	int rc;
  
	rc = sqlite3_open("familyGuy.db", &db);

	if( rc )
	{
		cout<<"Can't open database: "<<sqlite3_errmsg(db)<<"\n";
	} 
	else
	{
		cout<<"Open database successfully\n\n";
	}

	pSQL[0] = "create table myTable (FirstName varchar(30), LastName varchar(30), Age smallint, Hometown varchar(30), Job varchar(30))";

	pSQL[1] = "insert into myTable (FirstName, LastName, Age, Hometown, Job) values ('Peter', 'Griffin', 41, 'Quahog', 'Brewery')";
	
	pSQL[2] = "insert into myTable (FirstName, LastName, Age, Hometown, Job) values ('Lois', 'Griffin', 40, 'Newport', 'Piano Teacher')";
	
	pSQL[3] = "insert into myTable (FirstName, LastName, Age, Hometown, Job) values ('Joseph', 'Swanson', 39, 'Quahog', 'Police Officer')";
	
	pSQL[4] = "insert into myTable (FirstName, LastName, Age, Hometown, Job) values ('Glenn', 'Quagmire', 41, 'Quahog', 'Pilot')";

	pSQL[5] = "select * from myTable";
	
	pSQL[6] = "delete from myTable"; 

	pSQL[7] = "drop table myTable";

	for(int i = 0; i < STATEMENTS; i++)
	{
		rc = sqlite3_exec(db, pSQL[i], callback, 0, &zErrMsg);
		if( rc!=SQLITE_OK )
		{
			cout<<"SQL error: "<<sqlite3_errmsg(db)<<"\n";
			sqlite3_free(zErrMsg);
			break;
		}
	}

	sqlite3_close(db);

	return 0;
}