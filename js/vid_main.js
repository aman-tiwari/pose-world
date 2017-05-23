var localForage = require('localforage');
var p5 = require('p5');

var urls = [
  'ATMA_Yoga_Dance_Group_LHCC-J9ljbBm5r0A.json',
  'A_AP_Ferg_-_Shabba_Explicit_ft._A_AP_ROCKY-iXZxipry6kE.json',
  'A_AP_ROCKY_-_F_kin_Problems_ft._Drake_2_Chainz_Kendrick_Lamar-liZm1im2erU.json',
  'A_AP_Rocky_-_Multiply_feat._Juicy_J-5v6JUzxWoGw.json',
  'BROOKE_CANDY_DAS_ME_OFFICIAL_VIDEO-DKQCX9AslFg.json',
  'Britney_Spears_-_Toxic_Official_Video-LOZuxwVk7TU.json',
  'Christina_Aguilera_-_Dirrty_ft._Redman-4Rg3sAb8Id8.json',
  'Future_Codeine_Crazy_WSHH_Premiere_-_Official_Music_Video-7cDYYvOhKwg.json',
  'Karo_Swen_-_Pole_Dance_-_ARTWORK_1_Tha_Trickaz-rCRP-5om_3Y.json',
  'Kendrick_Lamar_-_Alright-Z-48u_uWMHY.json',
  'Kraftwerk_-_The_Robots-VXa9tXcMhXQ.json',
  'Magic_Mike_Raining_Men_Trailer_Official_2012_1080_HD_-_Channing_Tatum-hHGJ7n_S0Wk.json',
  'Magic_Mike_XXL_Official_Trailer-lMulcnyxzxQ.json',
  'Migos_-_Bad_and_Boujee_ft_Lil_Uzi_Vert_Official_Video-S-sJp1FfG7Q.json',
  'PSY_-_GANGNAM_STYLE_M_V-9bZkp7q19f0.json',
  'Pole_Dance_-_Nana_by_Trey_Songz-kCClY2e11rQ.json',
  'QT_-_Hey_QT-1MQUleX1PeA.json',
  'Skepta_featuring_JME_-_ThatsNotMe-dyONbqggasY.json',
  'Yoga_for_Core_Strength_Twists_-_30_min_Yoga_Flow-Vm2bXoqDPAM.json',
  'Young_Thug_-_Wyclef_Jean_Official_Video-_9L3j-lVLwk.json',
  'Yung_Lean_-_Blinded-AB5rd5JVBto.json'
];

var sources = [
  {
    'name': 'atma_yoga_dance',
    'url': 'jsons/ATMA_Yoga_Dance_Group_LHCC-J9ljbBm5r0A.json',
    'corner': [200, 100],
    'resize_to': 'None',
    'colour': [40, 25, 92],
    'speed': 1.5
  },
  {
    'name': 'shabba',
    'url': 'jsons/A_AP_Ferg_-_Shabba_Explicit_ft._A_AP_ROCKY-iXZxipry6kE.json',
    'corner': [400, 20],
    'resize_to': 'None',
    'colour': [10, 25, 12],
    'speed': 1.5
  },
  {
    'name': 'fucking_problems',
    'url':
        'jsons/A_AP_ROCKY_-_F_kin_Problems_ft._Drake_2_Chainz_Kendrick_Lamar-liZm1im2erU.json',
    'corner': [500, 40],
    'resize_to': 'None',
    'colour': [10, 25, 12],
    'speed': 1.5
  },
  {
    'name': 'multiply',
    'url': 'jsons/A_AP_Rocky_-_Multiply_feat._Juicy_J-5v6JUzxWoGw.json',
    'corner': [700, 10],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.1
  },
  {
    'name': 'das_me',
    'url': 'jsons/BROOKE_CANDY_DAS_ME_OFFICIAL_VIDEO-DKQCX9AslFg.json',
    'corner': [650, 200],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'toxic',
    'url': 'jsons/Britney_Spears_-_Toxic_Official_Video-LOZuxwVk7TU.json',
    'corner': [900, 80],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'dirrty',
    'url': 'jsons/Christina_Aguilera_-_Dirrty_ft._Redman-4Rg3sAb8Id8.json',
    'corner': [1300, 95],
    'resize_to': 'None',
    'colour': [95, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'codeine_crazy',
    'url':
        'jsons/Future_Codeine_Crazy_WSHH_Premiere_-_Official_Music_Video-7cDYYvOhKwg.json',
    'corner': [1320, 100],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'karo_pole_dance',
    'url':
        'jsons/Karo_Swen_-_Pole_Dance_-_ARTWORK_1_Tha_Trickaz-rCRP-5om_3Y.json',
    'corner': [1800, 1],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'kendrick_alright',
    'url': 'jsons/Kendrick_Lamar_-_Alright-Z-48u_uWMHY.json',
    'corner': [2300, 30],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'kraftwerk_robots',
    'url': 'jsons/Kraftwerk_-_The_Robots-VXa9tXcMhXQ.json',
    'corner': [2460, 90],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'magic_mike_raining_men',
    'url':
        'jsons/Magic_Mike_Raining_Men_Trailer_Official_2012_1080_HD_-_Channing_Tatum-hHGJ7n_S0Wk.json',
    'corner': [2790, 100],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'magic_mike_official',
    'url': 'jsons/Magic_Mike_XXL_Official_Trailer-lMulcnyxzxQ.json',
    'corner': [2530, 120],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'migos_bad_and_boujee',
    'url':
        'jsons/Migos_-_Bad_and_Boujee_ft_Lil_Uzi_Vert_Official_Video-S-sJp1FfG7Q.json',
    'corner': [2600, 90],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'psy_gangam_style',
    'url': 'jsons/PSY_-_GANGNAM_STYLE_M_V-9bZkp7q19f0.json',
    'corner': [2900, 120],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'nana_pole_dance',
    'url': 'jsons/Pole_Dance_-_Nana_by_Trey_Songz-kCClY2e11rQ.json',
    'corner': [3500, 20],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'hey_qt',
    'url': 'jsons/QT_-_Hey_QT-1MQUleX1PeA.json',
    'corner': [4200, 100],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'sketa_thats_not_me',
    'url': 'jsons/Skepta_featuring_JME_-_ThatsNotMe-dyONbqggasY.json',
    'corner': [3800, 90],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'yoga_for_core',
    'url':
        'jsons/Yoga_for_Core_Strength_Twists_-_30_min_Yoga_Flow-Vm2bXoqDPAM.json',
    'corner': [3670, 10],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 1.5
  },
  {
    'name': 'young_thug_wyclef',
    'url': 'jsons/Young_Thug_-_Wyclef_Jean_Official_Video-_9L3j-lVLwk.json',
    'corner': [3800, 30],
    'resize_to': 'None',
    'colour': [29, 23, 200],
    'speed': 1.5
  },
  {
    'name': 'yung_lean_blinded',
    'url': 'jsons/Yung_Lean_-_Blinded-AB5rd5JVBto.json',
    'corner': [3900, 90],
    'resize_to': 'None',
    'colour': [29, 23, 200],
    'speed': 1.5
  },
  {
    'name': 'leap_o_faith',
    'url': 'jsons/leap_o_faith.json',
    'corner': [1700, 3],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 2.0
  },
  {
    'name': 'worlds_best',
    'url': 'jsons/worlds_best.json',
    'corner': [30, 1],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 2.0
  },
  {
    'name': 'foreign_one',
    'url': 'jsons/foreign_one.json',
    'corner': [3600, 100],
    'resize_to': 'None',
    'colour': [29, 23, 98],
    'speed': 2.0
  }
];

var sizes = sources.map(x => x.corner[0]).sort();
var max_x = 4200;  // sizes[sizes.length - 1];

function fetch_cached(url) {
  return localForage.getItem(url).then(item => {
    if (item == null) {
      return window.fetch(url).then(res => res.json()).then(res => {
        console.log('caching', url);
        return localForage.setItem(url, res).then(_ => res);
      });
    } else {
      return item;
    }
  });
}

Promise.all(sources.map(x => x['url']).map(fetch_cached)).then(frames_jsons => {
  sources = sources.map((x, i) => {
    x.frames = frames_jsons[i].frames;
    console.log(x.name, x.frames.length)
    localStorage.setItem(x.name, x.frames);
    return x;
  });

  var sketch_fn = function(p) {

    var graphics = [];
    var updates = [];
    var scroll_offset = 0;

    p.setup = function() {
      p.createCanvas(1920, 480);
      p.background('white');
      p.frameRate(55);

      var mk_graphics = function(place, frames) {
        var g = p.createGraphics(640, 640);
        graphics.push({place: place, graphics: g});
        console.log(frames.length);
        var i = 0;

        var update = function() {
          i++;
          g.clear();


          if (frames[Math.floor(i / place.speed) % frames.length] == null)
            return;

          g.stroke.apply(g, place.colour);
          g.strokeWeight(15);

          var all_ppl =
              frames[Math.floor(i / place.speed) % frames.length]['found_ppl'];

          for (var skel of all_ppl) {
            var bones = Object.keys(skel);
            for (var bone_name of bones) {
              g.line(
                  skel[bone_name][0][1], skel[bone_name][0][0],
                  skel[bone_name][1][1], skel[bone_name][1][0])
            }
          }

          g.stroke.apply(g, [255, 255, 255]);
          g.strokeWeight(10);

          for (var skel of all_ppl) {
            var bones = Object.keys(skel);
            for (var bone_name of bones) {
              g.line(
                  skel[bone_name][0][1], skel[bone_name][0][0],
                  skel[bone_name][1][1], skel[bone_name][1][0])
            }
          }
        };

        updates.push(update);
      };

      sources.forEach(x => mk_graphics(x, x.frames))
    };


    p.draw = function() {
      p.background(255, 255, 255, 100);
      p.rect(-1, -1, 1919, 479);
      for (var j = 0; j < updates.length; j++) {
        var g = graphics[j];
        var update = updates[j];
        if (g.place.corner[0] - scroll_offset < 0) continue;
        update();
      }
      for (var g of graphics) {
        if (g.place.corner[0] - scroll_offset < 0) continue;
        p.image(
            g.graphics, g.place.corner[0] - scroll_offset, g.place.corner[1])
      }
      scroll_offset = (scroll_offset + 1) % max_x;
      console.log(scroll_offset);
    }

  };

  var sketch = new p5(sketch_fn);
});