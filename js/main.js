var fromWebSocket = require('most-w3msg').fromWebSocket;
var p5 = require('p5');

window.fetch('/places.json')
.then(res => res.json())
.then(PLACES => {

var SERV_IP = 'put_ip_address_here'
var SERV_URL = `ws://${SERV_IP}:8008`

function get_bone_stream(url, resize_to = 'None') {
    if(resize_to == undefined) {
        resize_to = 'None'
    }
    var raw = new WebSocket(`${SERV_URL}/bones?resize_to=${resize_to}&cam_url=${url}`)
    return fromWebSocket(raw)
}

var sketch_fn = function(p) {
    var graphics = [];
    p.setup = function() {
        p.createCanvas(1300, 1300);
        p.background('white')
        
        var places = Object.keys(PLACES)
        var k = 0;

        for(var place_name of places) {
            var bone_stream = get_bone_stream(PLACES[place_name]['url'],
                                        PLACES[place_name]['resize_to'])
            var g = p.createGraphics(PLACES[place_name]['resize_to'][0], 
                                     PLACES[place_name]['resize_to'][1]);
            graphics.push({
                place: PLACES[place_name],
                graphics: g
            })

            bone_stream.map(x => x.data)
                .map(JSON.parse)
                .forEach(((k, place_name, g) => (msg) => {
                    console.log(msg);
                    if(Math.random() > 0.3) g.clear();
                    var place = PLACES[place_name];
                    var stream_colour = place.colour;

                    var all_ppl = msg['found_ppl']
                    g.noFill()
                    g.stroke.apply(g, place.colour)
                    g.strokeWeight(2)

                    var cor_x = PLACES[place_name]['corner'][0];
                    var cor_y = PLACES[place_name]['corner'][1];

                    g.rect(3 , 3, PLACES[place_name]['resize_to'][0] - 3, PLACES[place_name]['resize_to'][1] - 3)
                    for(var skel of all_ppl) {
                        var bones = Object.keys(skel);
                        for(var bone_name of bones) {
                            g.line( skel[bone_name][0][1],
                                 skel[bone_name][0][0],
                                 skel[bone_name][1][1],
                                 skel[bone_name][1][0])
                        }
                    }
                    console.log(place_name);
                })(k, place_name, g))
            k++;
        }
    }

    p.draw = function () {
        p.background(15, 15, 20)
        for(var g of graphics) {
            p.image(g.graphics, 
                g.place.corner[0],
                g.place.corner[1]
            )
        }
    }

}

var sketch = new p5(sketch_fn);
});