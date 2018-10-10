Instructions indicated for PyEnchant (Python) with Tensorflow

Pre-requisites:

1. Download and ungzip GoogleNews-vectors-negative300.bin.gz from:

    https://github.com/mmihaltz/word2vec-GoogleNews-vectors

2. Run pattern removal script, clean_data.py: 

    `python clean_data.py`


3. Train the Neural Network: train.py (experiment could be either Trump or all)
 
    `python train.py --experiment=Trump`

     Once train.py is finished, a directory in 'run' has been created, which cointains the network parameters.

4. Test the Neural Network eval.py

    ```python eval.py --experiment=Trump```

Seperate Words by Part of Speech:
    - run 'before_pos.py' (make sure to change directory, cur_dir = "trump_final1/" --> cur_dir = "YOUR_DIRECTORY/"
    - run 'parts_of_speech.py' (make sure to change directory, directory = "trump_final1/" --> directory = "YOUR_DIRECTORY/"

Acronym & Abbreviations:
rb = real bodies
fb = fake bodies
fb_train = all fake bodies training dataset
fb_test = all fake bodies testing dataset
rb_train = all real bodies training dataset
rb_test = all real bodies testing dataset

File Explanations:
Within directory folder, there are several files.
    false_neg.txt - article is 'real', classified as 'fake'
    false_pos.txt - article is 'fake', classified as 'real'
    true_neg.txt - article is 'fake', classified as 'fake'
    true_pos.txt - article is 'real', classified as 'real'

* Data's directory holds all fake bodies and real bodies text files. It also holds 'news-data`.*
  Within 'news-data', holds `trump` text files and `email` text files:
        - no_trump_fb.txt and no_trump_rb.txt are the training files
        - trump_fb.txt and trump_rb.txt are the testing files
        - no_email_fb.txt and no_email_rb.txt are the training files
        - email_fb.txt and email_rb.txt are the testing files

  Also, within 'news-data,' holds fake and real text bodies:
        - fb_train and rb_train are the training files
        - fb_test and rb_test are the training files
