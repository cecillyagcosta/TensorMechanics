import os
import numpy as np
from skimage import io, color, measure, morphology
from PIL import Image

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1_HEATCORE_v4"

def run(indir, outdir, threshold_value, dpi=300):
    os.makedirs(outdir, exist_ok=True)

    def save_high_res_image(image_array, save_path, dpi):
        pil_image = Image.fromarray(image_array)
        pil_image.save(save_path, dpi=(dpi, dpi))

    def classify_image_by_heat(nucleus_intensity):
        if nucleus_intensity < 33.3:
            return None
        elif 33.3 <= nucleus_intensity < 100:
            return "muito_frio"
        elif 100 <= nucleus_intensity < 200:
            return "frio"
        elif 200 <= nucleus_intensity < 300:
            return "morno"
        elif 300 <= nucleus_intensity < 400:
            return "quente"

    def process_image(image_path, outdir, threshold_value):
        image = io.imread(image_path)

        if image.shape[2] == 4:
            image = image[:, :, :3]

        gray_image = color.rgb2gray(image)
        binary = gray_image > threshold_value
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            cropped_image = image[minr:maxr, minc:maxc]
            
            mean_intensity = gray_image[minr:maxr, minc:maxc].mean() * 400  # Heat based on average intensity
            heat_class = classify_image_by_heat(mean_intensity)
            
            if heat_class:
                class_outdir = os.path.join(outdir, heat_class)
                os.makedirs(class_outdir, exist_ok=True)
                
                # Save the classified image
                save_path = os.path.join(class_outdir, f'{base_name}_figura_{i}.png')
                save_high_res_image(cropped_image, save_path, dpi)
                
                # Log the position of the region
                print(f"Image: {base_name}_figura_{i}, Position: ({minr},{minc}), Heat: {heat_class}")

    for filename in os.listdir(indir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir, threshold_value / 400)

print("starting")
run(imdir, sampleoutdir, 33.3, dpi=1200)
print("run done")
