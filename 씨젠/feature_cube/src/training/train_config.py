# Common configuration
anatomy = 'colon'
diagnosis = ['D', 'M', 'N']
tile_size = 256
resolution_factor = 4
overlap = 8
ext = 'jpg'
limit_bounds = False
tiler_workers = 20
slide_distribution_path = f"./{anatomy}_slide_distribution.csv"
tile_dir = f'./tiles/{anatomy}'


# Step 1: Patch classifier using mixpatch
# patch_classifier_model_architecture = "efficientnet-b3"
# patch_classifier_data_dir = 'tiles'
# patch_classifier_batch_size = 512
# patch_classifier_num_epochs = 20
# patch_classifier_model_save_dir = 'models/3class/patch_classifier'
# patch_classifier_sl = True
# patch_classifier_lr = 0.01


# Step 2: Building the feature cube
feature_cube_model_architecture = "efficientnet-b3"
feature_cube_patch_classifier_path = f"./lossdiff/merged_{anatomy.lower()}_lossdiff_balanced.pkl"
feature_cube_path = f"./feature_cubes/{anatomy}"
lsize = 256
csize = 128
feature_cube_batch_size = 128

# Step 3: Slide classifier
slide_classifier_data_path = f"./feature_cubes/{anatomy}"
slide_classifier_saving_path = f'./models/{anatomy}/slide_classifier'
slide_classifier_batch_size = 64
slide_classifier_num_epochs = 100
slide_classifier_lr = 0.01
slide_classifier_dropout = 0.01
slide_classifier_sl = True

if anatomy.lower() == "stomach":
    slide_model_path = "./models/stomach/slide_classifier/20220207_124756/slide_classifier.pt"
else:
    slide_model_path = "./models/colon/slide_classifier/20220206_152615/slide_classifier.pt"
