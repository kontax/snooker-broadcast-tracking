var express = require('express');
var app = express();

require('./config')(app);
require('./routes')(app);
app.listen(8080);
console.log('App is running on http://localhost:8080');
