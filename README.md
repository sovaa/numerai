## Numerai Baseline Project

### About

This project is what has evolved from my participation in the numer.ai machine learning competitions. It's nothing more
than experimentation and thus it only contains some scripts and not a full-fledged project.

Feel free to use this project as a baseline to continue with! I only ask that if you do, release the source under GPLv3
as I have done, so improvements to it can help others learn more. If you only wish to copy snippets, feel free to do so
without publishing your work, but please refer to this repository if you.

Pull-requests are more than welcome, since I personally am not particularly skilled in the ML field, but wish to learn
more.

### Installation

Miniconda3 has been my goto for a while so I'll include a setup example for that:

    sovaa@stink ~ $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sovaa@stink ~ $ bash Miniconda3-latest-Linux-x86_64.sh # assuming defaults accepted and path added to .bashrc
    sovaa@stink ~ $ source ~/.bashrc

Clone the project:

    sovaa@stink ~ $ git clone https://github.com/sovaa/numerai.git
    sovaa@stink ~ $ cd numerai/

Create your environment and install the requirements:

    sovaa@stink ~/numerai $ conda create -n numerai python=3.5
    sovaa@stink ~/numerai $ source activate numerai
    (numerai) sovaa@stink ~/numerai $ pip install -r requirements.txt

Script assumes files called `data_train.csv` and `data_predict.csv`:
    
    (numerai) sovaa@stink ~/numerai $ mv numerai_training_data.csv data_train.csv
    (numerai) sovaa@stink ~/numerai $ mv numerai_tournament_data.csv data_predict.csv

### Model Description

The model is a 2 level stack; first level uses a couple of classifiers, which (as of writing) is:

* XGBoost 1,
* XGBoost 2,
* Random Forest
* Ada Boost (with Extra Randomized Trees as base classifier)

The level 2 blender is:

* XGBoost

Training of the first level is done by splitting the training data by era, and train each of the above L1 classifiers
on that era, then predict all the validation examples. When the iteration has finished, the resulting validation
predictions will thus be a much larger matrix than the original training data.

Preprocessing of this larger matrix is done by averaging the predictions per training ID, giving us the original number 
of training examples again. The features are then transformed by first transforming them into polynomial features, then
a 5 component PCA transformation.

Lastly, the L2 blender will then train on the training PCA data and predict on the prediction PCA data. 

### Example Run

Train and output the `tournament_results.csv` file:

    (numerai) sovaa@stink ~/numerai $ time python stacking.py

_After_ the `stacking.py` script has run, you'll have the output from level 1 of the stacker in these two files:

    l2_x_train.csv
    l2_y_train.csv
    l2_x_test.csv
    blend_x_train.csv
    blend_x_test.csv

The level 2 blender is e.g. LR/XGBoost; the X input for it is the `blend_x_train.csv` and `blend_x_test.csv`. These two
matrices are transformations of the `l2_x_*` files like this pseudo code describes:

    blend_x_train = PCA(polynomial_features(l2_x_train))

The reason why the `l2_x_*` files are saved at all is because the second script, `keras_for_l2.py`, used them as input
for a NN:

    (numerai) sovaa@stink ~/numerai $ time python keras_for_l2.py
