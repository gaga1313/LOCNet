import json
import numpy as np

with open('human_identifable_category_info.json', 'r') as file:
    data = json.load(file)

classes = {}
for key,value in data.items():
    classes[key] = np.random.choice(value['IN_category'])

with open('imagenet_sub_category.json', 'w') as file:
    json.dump(classes, file)

