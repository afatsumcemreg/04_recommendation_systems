######################################################################
# Content Based Filtering using the Metadata of Movies
######################################################################

######################################################################
# 1. Business Problem
######################################################################

# A newly established online movie viewing platform wants to make movie recommendations to its users.Since the login rate of users is very low, it cannot collect user habits. For this reason, it cannot develop product recommendations with the collaborative filtering method. but it knows which movies the users are watching from their tracks in the browser. It is requested to make movie recommendations based on this information.

######################################################################
# 2. Dataset Story
######################################################################

# The dataset contains basic information about 45000 movies. Within the scope of the application, it was worked with the 'overview' variable containing movie descriptions.

######################################################################
# 3. Creating the TF-IDF Matrix
######################################################################

# Goal: Developing a Content-Based Recommendation System

# Importing the libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

# Reading the dataset
df = pd.read_csv('datasets/movies_metadata.csv', low_memory=False, encoding='utf-8') # to close the DtypeWarning
df.head()
df.shape

# Selecting the variable 'overview'
df['overview'].head()
# 0    Led by Woody, Andy's toys live happily in his ...
# 1    When siblings Judy and Peter discover an encha...
# 2    A family wedding reignites the ancient feud be...
# 3    Cheated on, mistreated and stepped on, the wom...
# 4    Just when George Banks has recovered from his ...

# it is necessary to translate the textual expressions here into measurable mathematical expressions

# Use of TF-IDF Method

tfidf = TfidfVectorizer(stop_words='english')
# We want to exclude commonly used expressions from the dataset that have no measurement value and generate too many empty observations.

# Replace missings in the dataset with spaces
df['overview'] = df['overview'].fillna('')

# Converting the variable 'overview' by calling the TF-IDF method
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape  # (45466, 75827)
# Thus, there are 45466 thousand comments and a new variable was formed from 75827 word names.
# At the intersection of observations and variables, there are TF-IDF scores.
# Getting the names of the variables
tfidf.get_feature_names()
# Getting the TF-IDF scores
tfidf_matrix.toarray()

######################################################################
# 4. Cosine Similarity Calculation
######################################################################

# Goal: Creating the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape