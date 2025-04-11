import random
from openimages.download import download_images

labels = [
    'Cat', 'Dog', 'Car', 'Tree', 'Person', 'Bicycle', 'Chair', 'Laptop',
    'Book', 'Bottle', 'Bird', 'Flower', 'Airplane', 'Train', 'Boat',
    'Horse', 'Bus', 'Guitar', 'Sofa', 'Telephone', 'Motorcycle', 'Elephant',
    'Zebra', 'Pizza', 'Clock', 'Cup', 'TV', 'Frisbee', 'Backpack', 'Banana'
]

selected_labels = ["Dog", 
                   "Cat", 
                   "Polar bear", 
                   "Tiger", 
                   "Cheetah", 
                   "Elephant", 
                   "Kangaroo", 
                   "Dolphin", 
                   "Mouse", 
                   "Pig", 
                   "Teddy bear", 
                   "Koala",
                   "Panda",
                   "Rabbit",
                   "Hamster",
                   "Goat"]
print(f"Selected labels: {selected_labels}")

images_per_class = 5

for label in selected_labels:
    print(f"Downloading class: {label}")
    try:
        download_images(
            'data/',
            [label],
            limit=images_per_class,
            exclusions_path=None,
        )
    except Exception as e:
        print(f"❌ Error downloading {label}: {e}")
