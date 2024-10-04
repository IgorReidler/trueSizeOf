# Required libraries: Pillow, requests, numpy, plotly
# Install them using the following commands:
# pip install pillow requests numpy plotly

from PIL import Image, ImageOps, ImageChops
import requests
import numpy as np
from io import BytesIO
import random
import plotly.graph_objs as go
import plotly.io as pio

# Debug plot flag
debugPlotFlag = 1

# Plot images after background removal using Plotly
import plotly.express as px

def plot_image(image, title):
    image_np = np.array(image)
    fig = go.Figure()
    fig.add_trace(go.Image(z=image_np))
    fig.update_layout(title=title)
    pio.show(fig)

def plot_image_clean(image, title):
    image_np = np.array(image)
    fig = go.Figure()
    fig.add_trace(go.Image(z=image_np))
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    fig.update_layout(title=title, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    pio.show(fig)

# Plot 2D numpy array using Plotly
def plot_np(array, title):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=array, colorscale='gray'))
    fig.update_layout(title=title)
    pio.show(fig)

# Background removal function based on custom method
def bg_remove_igors(image, bg_threshold=10):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Calculate the grayscale image by floor(mean(r, g, b))
    gray_np = np.mean(image_np[:, :, :3], axis=2).astype(np.uint8)
    
    if debugPlotFlag:
        plot_np(gray_np, 'Grayscale Image')
    
    # Compute the most repeated value from 5x5 pixel rectangles at each of the 4 corners
    corners = [
        gray_np[:5, :5],  # Top-left
        gray_np[:5, -5:],  # Top-right
        gray_np[-5:, :5],  # Bottom-left
        gray_np[-5:, -5:]  # Bottom-right
    ]
    bg_values = [np.bincount(corner.flatten()).argmax() for corner in corners]
    bg_value = max(set(bg_values), key=bg_values.count)

    # Find top and bottom rows of the item using a matrix approach
    row_averages = np.mean(gray_np, axis=1)
    col_averages = np.mean(gray_np, axis=0)
    item_rows = np.where(np.absolute(row_averages - bg_value) > bg_threshold)[0]
    item_cols = np.where(np.absolute(col_averages - bg_value) > bg_threshold)[0]
    
    if len(item_rows) > 0:
        top_item_row = item_rows[0]
        bottom_item_row = item_rows[-1]
        gray_np = gray_np[top_item_row:bottom_item_row + 1, :]
        image_np = image_np[top_item_row:bottom_item_row + 1, :, :]
    
    if len(item_cols) > 0:
        left_item_col = item_cols[0]
        right_item_col = item_cols[-1]
        gray_np = gray_np[:, left_item_col:right_item_col + 1]
        image_np = image_np[:, left_item_col:right_item_col + 1, :]
    
    # Convert back to PIL image
    image = Image.fromarray(image_np, 'RGBA')
    return image

# Crop images to remove background from the top and sides
def crop_image(image):
    bg = Image.new(image.mode, image.size, (0, 0, 0, 0))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image


def process_images():
    try:
        # Generate a random number for debugging
        random_number = random.randint(1, 1000)

        # Image URLs and heights
        image1_url = 'https://assets.bose.com/content/dam/Bose_DAM/Web/consumer_electronics/global/products/speakers/s1_pro_system/product_silo_images/Chibi-silo1-1200x1022-17-BOSE-064-121417.psd/jcr:content/renditions/cq5dam.web.600.600.png'
        height1 = 330
        image2_url = 'https://media.guitarcenter.com/is/image/MMGS7/L35653000000000-00-600x600.jpg'
        height2 = 411

        # Download images
        image1 = Image.open(BytesIO(requests.get(image1_url).content)).convert('RGBA')
        image2 = Image.open(BytesIO(requests.get(image2_url).content)).convert('RGBA')

        if debugPlotFlag:
            plot_image(image1, "Original Image 1")

        # Remove background using custom method
        image1 = bg_remove_igors(image1, bg_threshold=0.05)
        image2 = bg_remove_igors(image2, bg_threshold=0.05)

        if debugPlotFlag:
            plot_image(image1, "Image 1 After Background Removal")
            plot_image(image2, "Image 2 After Background Removal")

        # Crop images to remove background from the top and sides
        # image1 = crop_image(image1)
        # image2 = crop_image(image2)

        # if debugPlotFlag:
        #     plot_image(image1, "Image 1 After Cropping")
        #     plot_image(image2, "Image 2 After Cropping")

        # Calculate scaling ratios
        ratio1 = height1 / image1.height
        ratio2 = height2 / image2.height

        # Scale images to correct relative size
        image1 = image1.resize((int(image1.width * ratio1), height1), Image.LANCZOS)
        image2 = image2.resize((int(image2.width * ratio2), height2), Image.LANCZOS)

        # Align bottoms by adjusting canvas height to fit both images
        max_height = max(image1.height, image2.height)
        image1_padded = Image.new('RGBA', (image1.width, max_height), (255, 255, 255, 0))
        image2_padded = Image.new('RGBA', (image2.width, max_height), (255, 255, 255, 0))

        image1_padded.paste(image1, (0, max_height - image1.height), image1)
        image2_padded.paste(image2, (0, max_height - image2.height), image2)

        if debugPlotFlag:
            plot_image(image1_padded, "Image 1 After Aligning Bottom")
            plot_image(image2_padded, "Image 2 After Aligning Bottom")

        # Create new image with both items side by side
        total_width = image1_padded.width + image2_padded.width
        result_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))

        # Paste the images next to each other
        result_image.paste(image1_padded, (0, 0), image1_padded)
        result_image.paste(image2_padded, (image1_padded.width, 0), image2_padded)

        plot_image_clean(result_image, "Final Combined Image")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    process_images()
