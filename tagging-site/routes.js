/**
 * This file contains the routes used within the web app. It loads the data
 * from the DB initialized in the init-db.js file, and displays one at a
 * time randomly.
 */

var sqlite3 = require('sqlite3').verbose();
var db = new sqlite3.Database('data/images.db');

// This select statement will grab a random image which has a tag that has not
// yet been set
var selectStatement = "\
    SELECT * \
    FROM images \
    WHERE tag1 IS NULL OR tag2 IS NULL OR tag3 IS NULL \
    ORDER BY RANDOM() \
    LIMIT 1";

module.exports = function(app) {
    
    // Main
    app.get('/', function(req, res) {

        var imageDetails = null;

        db.get(selectStatement, function(err, row) {
            imageDetails = row;
            res.render('home', { image: imageDetails });
        });
    });

    app.post('/yes', choose);
    app.post('/no', choose);

    function choose(req, res) {

        var choice = req.url.replace('/','');
        var image = req.body.image;
        var stmt = "\
            SELECT * \
            FROM images \
            WHERE path = '" + image + "';";

        db.get(stmt, function(err, row) {

            // Store the number of times the image has been tagged
            var tagCount = row.tag1 == null ? 0 : 1 + 
                           row.tag2 == null ? 0 : 1 + 
                           row.tag3 == null ? 0 : 1;

            // If untagged, just tag the choice
            if(tagCount == 0) {
                db.run("UPDATE images SET tag1 = ? WHERE path = ?", [ choice, image ]);
            }

            // If it's been tagged already, we need to see if the choice matches
            // the current tag, otherwise we reset the values
            else if(tagCount > 0 && row.tag1 == choice) {
                if(row.tag2 == null) {
                    db.run("UPDATE images SET tag2 = ? WHERE path = ?",
                            [ choice, image ]);
                } else {
                    db.run("UPDATE images SET tag3 = ? WHERE path = ?",
                            [ choice, image ]);
                }
            }

            // Otherwise we need to reset all the tags, as a mistake has been made
            else {
                db.run("UPDATE images SET tag1 = NULL, tag2 = NULL, tag3 = NULL WHERE path = ?",
                        [ image ]);
            }

            res.redirect('../');

        });
    }
};

//db.close();
