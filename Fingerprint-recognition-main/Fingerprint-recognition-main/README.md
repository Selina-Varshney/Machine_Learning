# Fingerprint-recognition
Fingerprint Recognition System:

The project is python based and uses OpenCV. 
For an authorised match it matches the datapoints using knn and flann based matcher, displaying that the user is authorised. 
For an unauthorised match, it prompts a message that the user 
is unauthorised.

Prerequisites:
Dataset downloaded from: https://www.kaggle.com/ruizgara/socofing
This would contain 2 folders:
Real- real fingerprints
Altered- blurred,faded,rotated versions of real dataset

Then run: main.py- the main python code.
Images- (authorised, no_entry) for displaying a prompt, have also been provided in this repository.

For authorized access:
Prompt:

![Verified_1](https://user-images.githubusercontent.com/99686864/236568506-c0f24953-89f2-467a-a2fe-5c1630c5f4e9.jpg)

Fingerprint Matching: 

![Verified_2](https://user-images.githubusercontent.com/99686864/236569598-0bb0d6dd-f1bd-45ba-8ad8-a22cd9f31d25.jpg)

Output:

![Verified_3](https://user-images.githubusercontent.com/99686864/236569638-a19430e7-b857-40d7-bc28-dc12f6a5b219.jpg)

For unauthorized access:
Prompt:

![Unauthorized_1](https://user-images.githubusercontent.com/99686864/236569757-f8e65407-0395-44a2-a614-3b8be1460b69.jpg)

Output:

![Unauthorized_2](https://user-images.githubusercontent.com/99686864/236569782-0e03040b-4218-4bf0-a832-e51149d7828d.jpg)

