# Required libraries: Flask, Pillow, requests, numpy, base64, plotly, rembg
# Install them using the following commands:
# pip install flask pillow requests numpy plotly rembg

from flask import Flask, render_template, request, send_file
from PIL import Image, ImageOps, ImageChops
import requests
import numpy as np
from io import BytesIO
import random
import base64
import plotly.graph_objs as go
import plotly.io as pio
from rembg import remove
import cv2

app = Flask(__name__)

# Debug plot flag
debugPlotFlag = 1

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html',
                           default_image1_url='https://assets.bose.com/content/dam/Bose_DAM/Web/consumer_electronics/global/products/speakers/s1_pro_system/product_silo_images/Chibi-silo1-1200x1022-17-BOSE-064-121417.psd/jcr:content/renditions/cq5dam.web.600.600.png',
                           default_image2_url='https://assets.bose.com/content/dam/Bose_DAM/Web/consumer_electronics/global/products/speakers/s1_pro_system/product_silo_images/Chibi-silo1-1200x1022-17-BOSE-064-121417.psd/jcr:content/renditions/cq5dam.web.600.600.png',
                           default_height1=450,
                           default_height2=900)

# Route to handle form submission
@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Generate a random number for debugging
        random_number = random.randint(1, 1000)

        # Get form data
        image1_url = request.form['image1_url']
        height1 = int(request.form['height1'])
        image2_url = request.form['image2_url']
        height2 = int(request.form['height2'])

        # Download images
        image1 = Image.open(BytesIO(requests.get(image1_url).content)).convert('RGBA')
        image2 = Image.open(BytesIO(requests.get(image2_url).content)).convert('RGBA')

        # Use rembg library for advanced background removal
        image1 = remove(image1, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)
        image2 = remove(image2, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)

        def bgremove1(myimage):
        
            # Blur to image to reduce noise
            myimage = cv2.GaussianBlur(myimage,(5,5), 0)
        
            # We bin the pixels. Result will be a value 1..5
            bins=np.array([0,51,102,153,204,255])
            myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
        
            # Create single channel greyscale for thresholding
            myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
        
            # Perform Otsu thresholding and extract the background.
            # We use Binary Threshold as we want to create an all white background
            ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
            # Convert black and white back into 3 channel greyscale
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        
            # Perform Otsu thresholding and extract the foreground.
            # We use TOZERO_INV as we want to keep some details of the foregorund
            ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  #Currently foreground is only a mask
            foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
        
            # Combine the background and foreground to obtain our final image
            finalimage = background+foreground
        
            return finalimage

        image1_cv2_remove=bgremove1(image1)
        
        plot_cv2_html = plot_image(image1_cv2_remove)
        # Plot images after background removal using Plotly
        def plot_image(image, title):
            image_np = np.array(image)
            fig = go.Figure()
            fig.add_trace(go.Image(z=image_np))
            fig.update_layout(title=title)
            return pio.to_html(fig, full_html=False)

        plot1_html = plot_image(image1, "Image 1 After Background Removal") if debugPlotFlag else ""
        plot2_html = plot_image(image2, "Image 2 After Background Removal") if debugPlotFlag else ""

        # Crop images to remove background from the top and sides
        def crop_image(image):
            bg = Image.new(image.mode, image.size, (255, 255, 255, 0))
            diff = ImageChops.difference(image, bg)
            bbox = diff.getbbox()
            if bbox:
                return image.crop(bbox)
            return image

        image1 = crop_image(image1)
        image2 = crop_image(image2)

        # Plot images after cropping using Plotly
        plot3_html = plot_image(image1, "Image 1 After Cropping") if debugPlotFlag else ""
        plot4_html = plot_image(image2, "Image 2 After Cropping") if debugPlotFlag else ""

        # Find the bottom row of the item separately from the background
        def find_bottom(image):
            image_np = np.array(image)
            alpha_channel = image_np[:, :, 3]
            non_empty_rows = np.where(alpha_channel > 0)[0]
            if len(non_empty_rows) > 0:
                return non_empty_rows[-1]
            return image.height - 1

        bottom1 = find_bottom(image1)
        bottom2 = find_bottom(image2)

        # Crop the images to align the bottom of the items
        image1 = image1.crop((0, 0, image1.width, bottom1 + 1))
        image2 = image2.crop((0, 0, image2.width, bottom2 + 1))

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

        # Plot images after aligning bottoms using Plotly
        plot5_html = plot_image(image1_padded, "Image 1 After Aligning Bottom") if debugPlotFlag else ""
        plot6_html = plot_image(image2_padded, "Image 2 After Aligning Bottom") if debugPlotFlag else ""

        # Create new image with both items side by side
        total_width = image1_padded.width + image2_padded.width
        result_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))

        # Paste the images next to each other
        result_image.paste(image1_padded, (0, 0), image1_padded)
        result_image.paste(image2_padded, (image1_padded.width, 0), image2_padded)

        # Save to a BytesIO object for sending as response
        output = BytesIO()
        result_image = result_image.convert('RGB')
        result_image.save(output, format='PNG')
        output.seek(0)

        # Encode image to base64 to display in HTML
        image_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

        return f"Random Number: {random_number}<br>{plot1_html}<br>{plot2_html}<br>{plot_cv2_html}<br>{plot3_html}<br>{plot4_html}<br>{plot5_html}<br>{plot6_html}<br><img src='data:image/png;base64,{image_base64}'>"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
