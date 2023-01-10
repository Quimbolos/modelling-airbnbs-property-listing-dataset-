# %% 
import os
import boto3
from PIL import Image

def process_images():

    def download_images():
        '''
        Downloads the images from an AWS bucket 

        Parameters
        ----------
        None

        Returns
        -------
        images: directory
            A directory containing all the images from the AWS bucket
        '''

        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket('myfirstbucketjbf') 
        for obj in bucket.objects.filter(Prefix = 'images'):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key) # save to same path

        return

    def resize_images():
        '''
        Iterates through the images folder to discard non-RBG images and resize them to the same height and width before saving them in the processed_images folder

        Parameters
        ----------
        None

        Returns
        -------
        processed_images: directory
            A directory containing all the processed images from the images folder
        '''

        base_dir = os.path.join(os.getcwd(),"images")

        rgb_file_paths = []

        # Get the RGB images file paths
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, f)
                    if os.path.isfile(file_path):
                        with Image.open(file_path) as img:
                            if img.mode == 'RGB':
                                rgb_file_paths.append(file_path)
        
        # Set the height of the smallest image as the height for all of the other images
        min_height = float('inf')
        for checked_file in rgb_file_paths:
            with Image.open(checked_file) as im:
                min_height = min(min_height, im.height)

        # Create the processed_images folder
        processed_images_path = os.path.join(os.getcwd(),"processed_images")
        if os.path.exists(processed_images_path) == False:
            os.makedirs(processed_images_path)

        # Resize all images to have the min_height whilst mantaining the aspect ratio
        for file_path in rgb_file_paths:
            with Image.open(file_path) as im:
                width, height = im.size
                new_height = min_height
                new_width = int(width * new_height / height)
                resized_im = im.resize((new_width, new_height))
                resized_im.save(os.path.join('processed_images', os.path.basename(file_path)))

        return

    download_images()

    resize_images()

    return


if __name__ == "__main__":
    process_images()


# %%

# %%
