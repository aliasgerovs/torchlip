import os
import kaggle

os.environ['KAGGLE_USERNAME'] = "aliasgerov"
os.environ['KAGGLE_KEY'] = "f8c3bba39668787321b36cd550f85dbe"

!kaggle datasets download -d adityajn105/flickr8k
!unzip flickr8k.zip