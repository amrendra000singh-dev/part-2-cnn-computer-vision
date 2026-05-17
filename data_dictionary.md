# Part 2 Dataset: Synthetic Manufacturing Defect Image Dataset

## Files
- `images/normal/`
- `images/scratch/`
- `images/dent/`
- `images/stain/`
- `labels.csv`

## Goal
Build a CNN model to classify product images into one of four classes.

## Classes
- `normal`: product surface without major defect
- `scratch`: product surface with scratch-like marks
- `dent`: product surface with circular dent-like marks
- `stain`: product surface with colored stain-like marks

## Suggested Student Tasks
- Load images from folders or from `labels.csv`
- Resize images to a fixed shape
- Normalize pixel values
- Train a CNN classifier
- Display predictions on sample images
- Discuss how this maps to manufacturing quality inspection
