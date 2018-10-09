Instructions indicated for PyEnchant (Python) with Tensorflow

Files + Cleaning:
    1. Download and ungzip GoogleNews-vectors-negative300.bin.gz
        (I downloaded from: https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

    2. Run pattern removal script, 'pattern_removal.py'
        * NOTE:
            - change txt_path to locate your directory and the file you want to run
                 (txt_path = '/Users/sofia/Documents/src/fakenews1/data/rb_test.txt')

Train and Test Neural Network:
    3. open train.py --> edit configuations --> enter parameters:
        python
        "/Users/sofia/Documents/src/fakenews1/train.py"
        --filter_sizes="3"
        --num_filters=128
        --positive_data_file="/Users/sofia/Documents/src/fakenews1/data/erb_train.txt"
        --negative_data_file="/Users/sofia/Documents/src/fakenews1/data/efb_train.txt"
    * NOTE:
     - Change path directory
     - positive_data_file = real news dataset
     - negative_data_file = fake news dataset

     4. Once train.py is finished, you should find a 'run' directory has been created.
        A checkpoint should also have been created within the 'run' directory (for example, a number should appear similiar to "1528206991")

     5. run eval.py --> edit configuations --> enter parameters:
        python
        "/Users/sofia/Documents/src/fakenews1/eval.py"
        --checkpoint_dir="/Users/sofia/Documents/src/fakenews1/runs/YOUR_NUMBER/checkpoints/"
        --eval_train
        --positive_data_file="/Users/sofia/Documents/src/fakenews1/data/erb_test.txt"
        --negative_data_file="/Users/sofia/Documents/src/fakenews1/data/efb_test.txt"
        --trigram_dir="all_final5/"

     * NOTE:
     - Change path directory
     - Change YOUR_NUMBER to the number that was created during training (i.e. runs/1528206991/checkpoints/)
     - CREATE a Python directory to store results (for example, mine was "all_final5")
     - positive_data_file = real news dataset
     - negative_data_file = fake news dataset

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