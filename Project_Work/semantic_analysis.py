# Importing libraries that are required for this capstone project.
  # Library details,
    # Pandas - for working with input dataset. 
    # spaCy  - general purpose natural language processing.

import pandas as pd
import spacy 
import os
from spacytextblob.spacytextblob import SpacyTextBlob

cwd = os.getcwd()
print(cwd)
for file in os.listdir(cwd):
    if file.startswith("amazon_product_reviews.csv"):
        loc =  os.path.join(cwd, file)


# Loading en_core_web_sm (small) from spacy library for assigning context-specific token vectors etc.
# Adding spacytextblob to pipline to enable sentiment analysis.

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Loading dataset to variable amazon_df.

amazon_df = pd.read_csv(loc,sep = ',', low_memory = False)
amazon_df.head()

# Details of the preprocessing steps - 
# Removing stop_words, punctuation from product review for further data analysis using preprocess_text function.
# Using spaCy's NLP library to perform tokenization.
# Concatenating tokens back to single string and return it.

def preprocess_text(text):
  doc = nlp(text)
  cleaned_text = [token.text.lower() for token in doc if not 
                token.is_stop and token.is_punct and token.text.strip()]
  return " ".join(cleaned_text)

# Analyzing sentiment polarity with Textblob using sentiment_analysis function.

def sentiment_analysis(new_reviews):
  for key,value in new_reviews.items():
    doc = nlp(value)
    print(f'\nReview: {key,value}')
    print(f'\nSentiment : {doc._.blob.sentiment}')

# Drop rows where reviews are missing and print the new total.

amazon_df = amazon_df.dropna(subset=['reviews.text'])
data_after_cleanup = amazon_df['reviews.text']
print(data_after_cleanup.shape)

# Apply preprocessing to clean the product reviews in dataset.

new_reviews = data_after_cleanup.apply(preprocess_text)
print(new_reviews)

# Evaluation of results.
# The polarity score is a float within the range [-1.0, 1.0] where -1.0 signifies a negative 
# sentiment, 1.0 signifies a positive sentiment, and values around 0 represent neutral sentiments. 
# Based on below results, it is observed that most of the selected reviews have positive sentiments.

# Selecting specific reviews for sentiment analysis.

reviews_data = amazon_df['reviews.text'].iloc[[10,30,50,100,200,400,800]]
sentiment_analysis(reviews_data)

# Similarity check between reviews with small model
for rev_pos in range (0,len(reviews_data)):
  if rev_pos < len(reviews_data)-2:

    item_loc1 = data_after_cleanup.iloc[rev_pos]
    item_loc2 = data_after_cleanup.iloc[rev_pos+1]

    item_rev1 = nlp(item_loc1)
    item_rev2 = nlp(item_loc2)
    
    small_model_sim_score = item_rev1.similarity(item_rev2)

    print(f"Review 1: {item_loc1}")
    print(f"Review 2: {item_loc2}")
    print(f'Similarity score of the two reviews: {round(small_model_sim_score,3)}')

# Loading en_core_web_sm (medium) from spacy library for better results
nlp = spacy.load('en_core_web_md')

# Review numbers will be received as input from user and will be compared against each other

item_1 = int(input('Enter the index of the first review for comparison: '))
item_2 = int(input('Enter the index of the second review for comparison: '))

total_reviews = len(data_after_cleanup)

# Input validation
if item_1 < 0 or item_2 < 0 or item_1 > total_reviews or item_2 > total_reviews:
   print(f"Invalid Index, Enter between 0 and {total_reviews}")
else:
    item1_review = data_after_cleanup.iloc[item_1]
    item2_review = data_after_cleanup.iloc[item_2]
    
    doc_a = nlp(item1_review)
    doc_b = nlp(item2_review)
    similarity_score = doc_a.similarity(doc_b)

    print(f"Review 1: {item1_review}")
    print(f"Review 2: {item2_review}")
    print(f'Similarity score of the two reviews: {round(similarity_score,3)}')