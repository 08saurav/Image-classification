***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 


## Dependencies

Pandas
Numpy
OpenCV
Seaborn
Keras
Matplotlib
Python
Scikit-learn
[TODO: Complete this section with the main dependencies and how to install them]

## Run Instructions

Must be runed from code folder

python3 train.py

python3 predict.py

If output == 0:
	Full Visibility
If output == 1:
	Partial Visibility
If output == 2:
	No Visibility
[TODO: Complete this section with how to run the project]

## Approach

After exploring the data, I came to know that only blue channel is contributing to features. Hence used blue channel of every image to train a CNN model with Adam optimizer. Number of classes are 3.

[TODO: Complete this section with a brief summary of the approach]

## Future Work

Will use Transfer Learning.
Use K-Fold Cross-Validation.
[TODO: Complete this section with a set of ideas for future work]