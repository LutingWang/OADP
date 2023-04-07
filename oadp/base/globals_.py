__all__ = [
    'Categories',
    'coco',
    'lvis',
    'Globals',
    'Store',
]

from typing import Iterable

import todd


class Store(metaclass=todd.StoreMeta):
    ODPS: bool
    DUMP: str


class Categories:

    def __init__(self, bases: Iterable[str], novels: Iterable[str]) -> None:
        self._bases = tuple(bases)
        self._novels = tuple(novels)

    @property
    def bases(self) -> tuple[str, ...]:
        return self._bases

    @property
    def novels(self) -> tuple[str, ...]:
        return self._novels

    @property
    def all_(self) -> tuple[str, ...]:
        return self._bases + self._novels

    @property
    def num_bases(self) -> int:
        return len(self._bases)

    @property
    def num_novels(self) -> int:
        return len(self._novels)

    @property
    def num_all(self) -> int:
        return len(self.all_)


class Globals(metaclass=todd.NonInstantiableMeta):
    """Entry point for global variables.

    Not to be confused with the global distillation branch.
    """
    categories: Categories
    training: bool


coco = Categories(
    bases=(
        'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
        'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
        'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
        'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
        'toothbrush'
    ),
    novels=(
        'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
        'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
        'sink', 'scissors'
    ),
)

lvis = Categories(
    bases=(
        'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke',
        'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award',
        'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag',
        'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid',
        'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow',
        'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
        'baseball_glove', 'basket', 'basketball', 'bat_(animal)', 'bath_mat',
        'bath_towel', 'bathrobe', 'bathtub', 'battery', 'bead', 'bean_curd',
        'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)',
        'beer_bottle', 'beer_can', 'bell', 'bell_pepper', 'belt',
        'belt_buckle', 'bench', 'beret', 'bib', 'bicycle', 'visor',
        'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
        'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 'blackberry',
        'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 'blouse',
        'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt',
        'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle',
        'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie',
        'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere',
        'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom',
        'brownie', 'brussels_sprouts', 'bucket', 'bull', 'bulldog',
        'bullet_train', 'bulletin_board', 'bullhorn', 'bun', 'bunk_bed',
        'buoy', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly',
        'button', 'cab_(taxi)', 'cabin_car', 'cabinet', 'cake', 'calculator',
        'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens',
        'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder',
        'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
        'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card',
        'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart',
        'carton', 'cash_register', 'cast', 'cat', 'cauliflower',
        'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
        'chair', 'chandelier', 'cherry', 'chicken_(animal)', 'chickpea',
        'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar',
        'chocolate_cake', 'choker', 'chopping_board', 'chopstick',
        'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern',
        'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock',
        'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat',
        'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_maker',
        'coffee_table', 'coffeepot', 'coin', 'colander', 'coleslaw',
        'pacifier', 'computer_keyboard', 'condiment', 'cone', 'control',
        'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkscrew',
        'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 'cowbell',
        'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon',
        'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown',
        'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
        'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
        'cupboard', 'cupcake', 'curtain', 'cushion', 'dartboard', 'deck_chair',
        'deer', 'dental_floss', 'desk', 'diaper', 'dining_table', 'dish',
        'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser',
        'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dolphin', 'domestic_ass',
        'doorknob', 'doormat', 'doughnut', 'drawer', 'underdrawers', 'dress',
        'dress_hat', 'dress_suit', 'dresser', 'drill',
        'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape',
        'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel',
        'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant',
        'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry',
        'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine',
        'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish',
        'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo',
        'flannel', 'flap', 'flashlight', 'flip-flop_(sandal)',
        'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
        'folding_chair', 'food_processor', 'football_(American)', 'footstool',
        'fork', 'forklift', 'freight_car', 'French_toast', 'freshener',
        'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'garbage_truck',
        'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda',
        'gift_wrap', 'ginger', 'giraffe', 'cincture',
        'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
        'golf_club', 'golfcart', 'goose', 'grape', 'grater', 'gravestone',
        'green_bean', 'green_onion', 'grill', 'grizzly', 'grocery_bag',
        'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'ham',
        'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer',
        'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil',
        'headband', 'headboard', 'headlight', 'headscarf',
        'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
        'highchair', 'hinge', 'hog', 'home_plate_(baseball)', 'honey',
        'fume_hood', 'hook', 'horse', 'hose', 'hot_sauce', 'hummingbird',
        'polar_bear', 'icecream', 'ice_maker', 'igniter', 'iPod',
        'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean',
        'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak',
        'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 'kitten',
        'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle',
        'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern',
        'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego',
        'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket',
        'lightbulb', 'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop',
        'speaker_(stero_equipment)', 'loveseat', 'magazine', 'magnet',
        'mail_slot', 'mailbox_(at_home)', 'mandarin_orange', 'manger',
        'manhole', 'map', 'marker', 'mashed_potato', 'mask', 'mast',
        'mat_(gym_equipment)', 'mattress', 'measuring_cup', 'measuring_stick',
        'meatball', 'medicine', 'melon', 'microphone', 'microwave_oven',
        'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money',
        'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
        'motor_scooter', 'motorcycle', 'mound_(baseball)',
        'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
        'musical_instrument', 'napkin', 'necklace', 'necktie', 'needle',
        'nest', 'newspaper', 'newsstand', 'nightshirt',
        'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'oar',
        'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice',
        'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet',
        'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
        'palette', 'pan_(for_cooking)', 'pancake', 'paper_plate',
        'paper_towel', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)',
        'passport', 'pastry', 'pea_(food)', 'peach', 'peanut_butter', 'pear',
        'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'pen', 'pencil',
        'penguin', 'pepper', 'pepper_mill', 'perfume', 'person', 'pet',
        'pew_(church_bench)', 'phonograph_record', 'piano', 'pickle',
        'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 'pinecone',
        'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pizza',
        'place_mat', 'plate', 'platter', 'pliers', 'pocketknife',
        'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony',
        'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot',
        'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel',
        'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector',
        'propeller', 'pumpkin', 'puppy', 'quilt', 'rabbit', 'racket',
        'radiator', 'radio_receiver', 'radish', 'raft', 'raincoat',
        'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)',
        'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector',
        'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe',
        'rocking_chair', 'rolling_pin', 'router_(computer_equipment)',
        'rubber_band', 'runner_(carpet)', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'sail',
        'salad', 'salami', 'salmon_(fish)', 'salsa', 'saltshaker',
        'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage',
        'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors',
        'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird',
        'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
        'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag',
        'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel',
        'shower_head', 'shower_curtain', 'signboard', 'silo', 'sink',
        'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
        'skirt', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowboard',
        'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa',
        'solar_array', 'soup', 'soupspoon', 'sour_cream', 'spatula',
        'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear',
        'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish',
        'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 'step_stool',
        'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign',
        'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)',
        'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)',
        'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop',
        'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato',
        'swimsuit', 'sword', 'table', 'table_lamp', 'tablecloth', 'tag',
        'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)',
        'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp',
        'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot',
        'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole',
        'television_camera', 'television_set', 'tennis_ball', 'tennis_racket',
        'thermometer', 'thermos_bottle', 'thermostat', 'thread', 'thumbtack',
        'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel',
        'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet',
        'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
        'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
        'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
        'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray',
        'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban',
        'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter',
        'umbrella', 'underwear', 'urinal', 'urn', 'vacuum_cleaner', 'vase',
        'vending_machine', 'vent', 'vest', 'videotape', 'volleyball', 'waffle',
        'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket',
        'wallet', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler',
        'water_faucet', 'water_jug', 'water_scooter', 'water_ski',
        'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam',
        'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair',
        'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill',
        'window_box_(for_plants)', 'windshield_wiper', 'windsock',
        'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)',
        'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet',
        'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'
    ),
    novels=(
        'applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor',
        'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo',
        'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball',
        'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag',
        'blimp', 'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark',
        'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball',
        'boxing_glove', 'brass_plaque', 'breechcloth', 'broach', 'bubble_gum',
        'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana',
        'locker', 'candy_bar', 'canteen', 'elevator_car', 'car_battery',
        'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail',
        'chaise_longue', 'chalice', 'chap', 'checkbook', 'checkerboard',
        'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk',
        'chocolate_mousse', 'cider', 'cigar_box', 'clarinet',
        'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)',
        'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil',
        'coloring_material', 'combination_lock', 'comic_book', 'compass',
        'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil',
        'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat',
        'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler',
        'curling_iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian',
        'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux',
        'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse', 'dove',
        'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan',
        'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot',
        'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)',
        'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet',
        'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask',
        'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla',
        'gourd', 'gravy_boat', 'griddle', 'grits', 'halter_top', 'hamper',
        'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium',
        'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey_stick', 'hookah',
        'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat',
        'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'jelly_bean',
        'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table',
        'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat',
        'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade',
        'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard',
        'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher',
        'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake',
        'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile',
        'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)',
        'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)',
        'pantyhose', 'papaya', 'paperback_book', 'paperweight', 'parchment',
        'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard',
        'pencil_box', 'pencil_sharpener', 'pendulum', 'pennant',
        'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank',
        'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe',
        'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume',
        'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding',
        'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet',
        'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat',
        'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate',
        'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate',
        'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone',
        'scarecrow', 'scraper', 'seaplane', 'sharpener', 'Sharpie',
        'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert',
        'shot_glass', 'shower_cap', 'shredder_(for_paper)', 'skullcap',
        'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero',
        'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
        'spear', 'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife',
        'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus',
        'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe',
        'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco',
        'tambourine', 'army_tank', 'telephoto_lens', 'tequila', 'thimble',
        'trampoline', 'trench_coat', 'triangle_(musical_instrument)',
        'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar',
        'violin', 'vodka', 'vulture', 'waffle_iron', 'walrus', 'wardrobe',
        'washbasin', 'water_heater', 'water_gun', 'wolf'
    ),
)
