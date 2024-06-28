from PIL import Image, ImageDraw

"""
Visualization utils
"""
def save_images_to_disk(img, path='images/', boxes=None, label=None, instance=False):
    """
    Function to save an image to disk with a bounding box and label
    """
    if not isinstance(img, Image.Image):
        img1 = Image.fromarray(img).copy()
    else:
        img1 = img.copy()

    if instance:
        img1.save(path+'instance.jpg')
        return

    if boxes is None or boxes == []:
        img1.save(path+'observation.jpg')
        return 
    
    draw = ImageDraw.Draw(img1)
    for i, box1 in enumerate(boxes):
        box = box1[0]
        color = 'red'
        draw.rectangle(box, outline=color, width=5)
        if label is not None:
            draw.text((box[0], box[1] - 5), label, fill='white')
    img1.save(path+'detection.jpg')
    return img1

def covered_map_percentage(fog_of_war):
    """
    Function to calculate the percentage of the map 
    covered by the agent
    """
    return (fog_of_war.sum() / fog_of_war.size) * 100