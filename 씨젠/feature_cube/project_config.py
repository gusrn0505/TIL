# Database Connection Configuration Start
mariadb = {
    "host": "219.252.39.14",
    "user": "root",
    "password": "seegeneai2020",
    "db": "digital_pathology",
}

# Test Configuration Start
anatomy = "Colon"
patch_classifier_path = f"./lossdiff/merged_{anatomy.lower()}_lossdiff_balanced.pkl"

# Tiling config
use_csv_file = True
slide_path = './test/'
slide_distribution_path = f"./{anatomy.lower()}_slide_distribution.csv"
tile_dir = "./temp/tiles"
tile_size = 256
tile_ext = "jpg"
resolution_factor = 4  # Zoom level = 16, most of the time
overlap = 8
use_filter = True
limit_bounds = False
delete_tile_dir = True  # Delete temp directory
save_patches_threshold = 0.1

# Step 1 - Binary classifier
step1_batch_size = 128

# Step 2 - Feature cube
feature_cube_path = "./temp/feature_cubes"
lsize = 256
csize = 128
step2_batch_size = 128

# Step 3 - Slide classifier
if anatomy.lower() == "stomach":
    slide_model_path = "./models/stomach/slide_classifier/20220207_124756/slide_classifier.pt"
elif anatomy.lower() == "colon":
    slide_model_path = "./models/colon/slide_classifier/20220206_152615/slide_classifier.pt"
