var sqlite3 = require('sqlite3').verbose();
var db = new sqlite3.Database('data/images.db');

var select = "\
    SELECT * \
    FROM images \
    WHERE tag1 IS NULL OR tag2 IS NULL OR tag3 IS NULL \
    ORDER BY RANDOM() \
    LIMIT 1";

db.get(select, function(err, row) {
    console.log(row);
});

db.each("SELECT * FROM images WHERE path = ? OR path = ?", ["00049-36.jpg", "00051-01.jpg" ], function(err, row) {
    console.log(row);
});

db.close();
