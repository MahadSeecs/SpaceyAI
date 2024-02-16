from PIL import Image
import os

def resize_images_in_folder(input_folder, output_folder, new_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each subfolder in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)

        # Create output subfolder if it doesn't exist
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # Loop through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            input_path = os.path.join(subfolder_path, filename)
            output_path = os.path.join(output_subfolder_path, filename)

            # Open and resize the image
            with Image.open(input_path) as img:
                # Resize the image while retaining the aspect ratio
                print(output_subfolder_path,filename)
                img.thumbnail(new_size)
                left_margin = (img.width - new_size[0]) // 2
                top_margin = (img.height - new_size[1]) // 2
                right_margin = left_margin + new_size[0]
                bottom_margin = top_margin + new_size[1]
                img = img.crop((left_margin, top_margin, right_margin, bottom_margin))
                # Save the resized image to the output folder
                img.save(output_path)

if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "planets"
    output_folder = "processed_ds"
    size = 224

    # Set the desired size for the images (width, height)
    new_size = (size,size)  # Adjust as needed

    # Resize the images and save them to the output folder
    resize_images_in_folder(input_folder, output_folder, new_size)
