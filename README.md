# Running the code from scratch

0. Pre-requisites: Python3 + packages: nltk, numpy, sklearn, Tensorflow (tested in version 1.12.0rc0)

1. Download and ungzip GoogleNews-vectors-negative300.bin.gz. Save the uncompressed GoogleNews-vectors-negative300.bin in the root directory of the repository (same directory as train.py). You can get the file here:

    https://github.com/mmihaltz/word2vec-GoogleNews-vectors
    

2. Run pattern removal script, clean_data.py: 

    `python clean_data.py`


3. Train the Neural Network: train.py (experiment could be either Trump or all)
 
    `python train.py --experiment=Trump`

     Stop the training when the validation accuracy does not increase anymore. The validation accuracy is displayed every 100 training steps. A directory in 'run' that cointains the network parameters is created.

4. Test the Neural Network eval.py

    ```python eval.py --experiment=Trump```

5. Get the most relevant patterns for each article:

     ```python get_patterns.py --experiment=Trump```
     
6. Display the most relevant patters accross all the dataset by parts of speech:
    
    ```python parts_of_speech.py --experiment=Trump```
    
    
# Dataset

* Data's directory holds all fake bodies and real bodies text files. It also holds 'news-data`.*
  Within 'news-data', holds `trump` text files and `email` text files:
        - no_trump_fb.txt and no_trump_rb.txt are the training files
        - trump_fb.txt and trump_rb.txt are the testing files
        - no_email_fb.txt and no_email_rb.txt are the training files
        - email_fb.txt and email_rb.txt are the testing files

  Also, within 'news-data,' holds fake and real text bodies:
        - fb_train and rb_train are the training files
        - fb_test and rb_test are the training files
