A description of the dataset used - 
===================================
Dataset contains Amazon product details such as 
      *  Product ID           *  product description   *  asins                   *  brand
      *  categories           *  keys                  *  manufacturer            *  reviews date added
      *  reviews date seen    *  reviews did purchase  *  reviews do recommend    *  reviews ID
      *  reviews num helpful  *  reviews rating        *  reviews source URLs     *  reviews text
      *  reviews title        *  reviews user city     *  reviews user province   *  reviews username  


Details of the preprocessing steps -
====================================
      *  The program utilizes the preprocess_text function, which removes stop words and punctuation from product reviews to facilitate further data analysis. 
      *  It employs spaCy's NLP library for tokenization, generating a list of lemmatized tokens from the processed text. 
      *  Subsequently, it concatenates these tokens into a single string before returning it.

Evaluation of results -
=======================
      *  The polarity score is a float within the range [-1.0, 1.0] where -1.0 signifies a negative sentiment, 1.0 signifies a positive sentiment and values around 0 represent neutral sentiments. 
      *  Based on below results, it is observed that most of the selected reviews have positive sentiments.
      *  Subjectivity lies between [0,1] and Subjectivity quantifies the amount of personal opinion and factual information contained in the text. 
      *  The higher subjectivity means that the text contains personal opinion rather than factual information. Based on this, most of the reviews were biased reviews than actual review of the product.
