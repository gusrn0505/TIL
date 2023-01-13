anatomy = 'colon'
assert anatomy in ["stomach", "colon"], f"Anatomy = {anatomy} not supported"
diagnosis = ['D', 'M', 'N']

slide_classifier_data_path = f"./feature_cubes/{anatomy}"

# if anatomy == "stomach":
#     slide_model_path = "./models/stomach/before_update/slide_classifier.pth"
# else:
#     slide_model_path = "./models/colon/before_update/slide_classifier.pth"

if anatomy.lower() == "stomach":
    slide_model_path = "./models/stomach/slide_classifier/20220207_124756/slide_classifier.pt"
elif anatomy.lower() == "colon":
    slide_model_path = "./models/colon/slide_classifier/20220412_210218/slide_classifier.pt"
