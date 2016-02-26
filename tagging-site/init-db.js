/**
 * This script creates the database used to store the path of the images and 
 * then inserts the data into it.
 */

var sqlite3 = require('sqlite3').verbose();
var fs = require('fs');
var db = new sqlite3.Database('data/images.db');

// The SQL code used to create the table when being run first. The path of the
// image is used as a Primary Key as we only want each image to exist once.
var tableInit = "\
CREATE TABLE IF NOT EXISTS images (\
    path varchar(100) PRIMARY KEY,\
    tag1 varchar(30),\
    tag2 varchar(30),\
    tag3 varchar(30)\
)";

// Get the images from the filesystem
var images = fs.readdirSync(__dirname + '/public/images');

// Initialize the database and insert the data
db.serialize(function() {

    db.run(tableInit);

    var insert = db.prepare("INSERT OR IGNORE INTO images(path) VALUES(?)");

    images.forEach(function(image) {
        insert.run(image);
        //db.run("INSERT OR IGNORE INTO images(path) VALUES ('" + image + "');");
    });

    insert.finalize();

    db.each("SELECT path FROM images", function(err, row) {
        console.log(row.path);
    });
});

db.close();
