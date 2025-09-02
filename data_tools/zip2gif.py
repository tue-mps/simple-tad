import sys

from data_tools.vis_utils import create_gif_from_zip




if __name__ == '__main__':
    _ = "mrIFkGZbNjc_003577"
    _ = "HaC3LrJiTmQ_006147"
    zip_filename = "/mnt/experiments/sorlova/datasets/DoTA_refined/frames/HaC3LrJiTmQ_006147/images.zip"
    output_filename = "HaC3LrJiTmQ_006147.gif"
    image_extension = ".jpg"
    create_gif_from_zip(zip_filename, image_extension, output_filename)

