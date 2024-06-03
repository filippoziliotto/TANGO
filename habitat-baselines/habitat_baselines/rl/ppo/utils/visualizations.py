from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import random

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

def generate_unique_color():
    """Generate a random color."""
    return tuple(random.randint(0, 255) for _ in range(3))

def overlay_segmentation(image, segmentation_outputs, save=True):
    """
    Overlay segmentation mask on the image
    save the image into scene_segmentation.jpg
    """
    overlay_image = image.copy()
    assert image.shape[:2] == segmentation_outputs[0]['mask'].shape, "Image and overlay image must have the same height"
    
    # Define a color map for different categories (for simplicity, we'll define a few colors)
    color_map = {}
    
    # Iterate over the segmentation outputs
    for output in segmentation_outputs:
        mask = output['mask']
        category = output['category']
        
        # Assign a unique color to each category
        if category not in color_map:
            color_map[category] = generate_unique_color()
        color = color_map[category]
        
        # Create a colored mask
        colored_mask = np.zeros_like(image)
        for i in range(3):  # Apply the same mask to all three channels
            colored_mask[:, :, i] = mask * color[i]
        
        # Overlay the colored mask on the image
        overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask.astype(np.uint8), 0.5, 0)

        # Find the center of the mask for placing the label
        mask_indices = np.where(mask == 1)
        if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            cv2.putText(overlay_image, category, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Create a legend
    legend_image = np.zeros((image.shape[0], 200, 3), dtype=np.uint8) + 255  # White background for the legend
    y_offset = 20
    for category, color in color_map.items():
        cv2.putText(legend_image, category, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(legend_image, (150, y_offset - 15), (180, y_offset + 5), color, -1)
        y_offset += 30
    
    # Combine the legend with the image
    combined_image = np.hstack((overlay_image, legend_image))

    # Save the resulting image if save is True
    if save:
        output_path = 'images/scene_segmentation.jpg'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay_image)

    return overlay_image