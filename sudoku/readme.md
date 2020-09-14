# Sudoku solver for aircowd blitz challenge

## Prerequsites

1. pytorch 1.6
2. pytorch-lightning
3. python 3.8
4. cv2
5. numpy
6. pandas

## How to use

1. Copy folder content in your directory
2. Download and unpack challenge data in "data" folder in your directory
3. Launch train_classifier.py to train model. Also you can skip this step and use pretrained model from "models" directory
4. Launch recognize_numbers_and_solve.py This script recognize numbers and store these predictions into pandas dataframe
5. Launch jupyter notebook "sudoku_solver.ipynb". File "test_labels_preds.csv" in data folder will be final solution.