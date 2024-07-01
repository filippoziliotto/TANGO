refined_names = {
    "tv_monitor": "tv",
    "toilet": "toilet seat",
    "plant": "plant",
    "sofa": "couch",
    "bed": "bed",
    "chair": "chair",
}

class_names_coco = {
    # ObjectNav
    "tv": "tv",
    "toilet": "toilet seat",
    "potted_plant": "plant",
    "couch": "couch",
    "bed": "bed",
    "chair": "chair",
    "couch": "sofa",
    # EQA
    'microwave': 'microwave',
    'refrigerator': 'refrigerator',
    'oven': 'oven',
    'toaster': 'toaster',
    'sink': 'sink',
    'refrigerator': 'fridge',
    'door': 'door',
    'plant': 'plant',
}

desired_classes_ids = {
    # ObjectNav
    62: "chair",
    63: "couch",
    64: "potted_plant",
    65: "bed",
    70: "toilet",
    72: "tv",
    # EQA
    71: "door",
    79: "oven",
    78: "microwave",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
}

stoi_eqa = {
'<unk>':0,
'black':11142,
'brown':11496,
'off-white':11367,
'white':11495,
'blue':11494,
'tan':11413,
'grey':10542,
'slate grey': 9584,
'silver':11046,
'green':11352,
'yellow green':3479,
'red brown':7712,
'yellow pink':1374,
'orange yellow':1234,
'bathroom':11462,
'kitchen':11493,
'lounge':11421,
'spa':2047,
'bedroom':11097,
'living room':10734,
'family room':10826,
'light blue':5947,
'tv room':3004,
'closet':11342,
'laundry room':11199,
'olive green':11115,
'foyer':4739,
'hallway':10388,
'dining room':10519,
'purple pink':7510,
'red':10915,
'purple':10636,
'yellow':10223,
'office':11243,
}

rooms_eqa = [
    'living room','family room','tv room','closet','laundry room',
    'hallway','dining room','office','bathroom','foyer','kitchen',
    'lounge','spa','bedroom', 'rec room',
]

colors_eqa = [
    'black','brown','off-white','white','blue','tan','grey',
    'slate grey','silver','green','yellow green','red brown',
    'yellow pink','orange yellow','light blue','olive green',
    'purple pink','red','purple','yellow',
]

roomcls_labels = {
    'living room': 'living room',
    'family room': 'living room',
    'tv room': 'living room',
    'rec room': 'living room',
    # 'closet': 'closet',
    'laundry room': 'laundry room',
    # 'hallway': 'hallway',
    'dining room': 'dining room',
    'bathroom': 'bathroom',
    'kitchen': 'kitchen',
    'bedroom': 'bedroom',
    'lounge': 'living room',
    # 'spa': 'spa',
    'office room': 'living room',
    'foyer': 'living room'
}

compact_labels = {
    'Bathroom': 'bathroom',
    'Bedroom': 'bedroom',
    'DinningRoom': 'dining room',
    'Kitchen': 'kitchen',
    'Laundry room': 'laundry room',
    'Livingroom': 'living room',
}

eqa_objects = {
    # 'towel': 'towel',
    # 'tv stand': 'tv',
    'tv': 'tv',
    'sofa': 'couch',
    'bed': 'bed',
    # 'shelving': 'shelving',
    'toaster': 'toaster',
    # 'clothes dryer': 'clothes dryer',
    'chair': 'chair',
    # 'fireplace': 'fireplace',
    # 'curtain': 'curtain',
    'sink': 'sink',
    'plant': 'potted plant',
    # 'counter': 'counter',
    'fridge': 'refrigerator',
    'door': 'door',
    # 'stove': 'stove',
    # 'picture': 'picture',
    # 'wardrobe': 'wardrobe',
    'microwave': 'microwave',
    'refrigerator': 'refrigerator',
    'oven': 'oven',
}