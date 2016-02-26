/**
 * Some configuration settings for the app.
 */

var handlebars = require('express-handlebars'),
    express = require('express'),
    bodyParser = require('body-parser');

module.exports = function(app) {

    app.engine('html', handlebars({
        defaultLayout: 'main',
        extname: ".html",
        layoutsDir: __dirname + '/views/layouts'
    }));

    app.set('view engine', 'html');
    app.set('views', __dirname + '/views');
    app.use(express.static(__dirname + '/public'));
    //app.use(express.urlencoded());
    app.use(bodyParser.urlencoded({ extended: false }));
};
