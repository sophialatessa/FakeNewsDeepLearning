This is the code to reproduce the results in the paper:

"Opening the Black-box of Deep Learning Fake News Detectors"

# Running the code from scratch

0. Pre-requisites: Python3 + packages: nltk, numpy, sklearn, Tensorflow (tested in version 1.12.0rc0)

1. Download and ungzip GoogleNews-vectors-negative300.bin.gz. Save the uncompressed GoogleNews-vectors-negative300.bin in the root directory of the repository (same directory as train.py). You can get the file here:

    https://github.com/mmihaltz/word2vec-GoogleNews-vectors
    

2. Run pattern removal script, clean_data.py: 
 ```
python clean_data.py
 ```

3. Train the Neural Network: train.py (experiment could be either Trump or all)
```
python train.py --experiment=Trump
```
Stop the training when the validation accuracy does not increase anymore. The validation accuracy is displayed every 100 training steps. A directory in 'run' that cointains the network parameters is created.

4. Test the Neural Network eval.py
```
python eval.py --experiment=Trump
```

5. Get the most relevant patterns for each article:
```
python get_patterns.py --experiment=Trump
```
     
6. Display the most relevant patters accross all the dataset by parts of speech:
```
python parts_of_speech.py --experiment=Trump
```
    
    
# Dataset

The dataset consists on the [Fake News Dataset by Kaggle collected by the BS detector](https://www.kaggle.com/mrisdal/fake-news) (7,401 articles) + articles collected from "The Guardian" and "The New York Times" (8,999 articles).

In the data directory it can be found:

- data/raw: the original articles
   
- data/processed: the articles after removing words that are not in the English dictionary via [PyEnchant](https://github.com/rfk/pyenchant)
   
- data/clean: the articles after cleaning advertisements and announcements, punctuations, etc. This is the data before going to the detector.
