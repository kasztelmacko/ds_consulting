import nltk
# download nltk if its not yet downloaded on your pc
# when downloaded just press exit "X" button
nltk.download()

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import etl
import re
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    Preprocesses the text by removing special characters, punctuation, and stopwords,
    and lemmatizing the words.

    Parameters:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Remove special characters, punctuation, and numbers
    text_clean = re.sub("[^a-zA-Z0-9 \n]", "", text)
    # Replace multiple spaces with a single space
    text_clean = re.sub(" +", " ", text_clean)
    # Remove punctuation
    text_clean = re.sub(r"[^\w\s]","", text_clean)
    # Remove digits
    text_clean = re.sub("\d", "", text_clean)
    # Convert the text to lowercase
    text_clean = text_clean.lower()
    # Tokenize the text into words
    word_tokens = word_tokenize(text_clean)
    # Remove English stopwords
    stop_words = set(stopwords.words("english")) 
    word_tokens = [w for w in word_tokens if w not in stop_words]
    # Lemmatize the words
    word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
    # Remove very short tokens
    word_tokens = [w for w in word_tokens if len(w) > 1]
    # Join the preprocessed tokens back into a string
    text_clean = " ".join(word_tokens)
        
    return text_clean


def filter_tokens(tokens):
    """
    Filters out undesired tokens.

    Parameters:
        tokens (list): List of tokens.

    Returns:
        list: Filtered list of tokens.
    """
    tokens_to_drop = ["aa", "ab", "aacx", "ba", "british", "airway"]
    return [token for token in tokens if token.lower() not in tokens_to_drop]

def create_bigram(row):
    """
    Creates bigrams from a row in the DataFrame.

    Parameters:
        row (Series): A row in the DataFrame.

    Returns:
        str: A bigram.
    """
    if row["Type"] == "Bigram":
        return row["Word1"] + "+" + row["Word2"]
    else:
        return ""

def create_trigram(row):
    """
    Creates trigrams from a row in the DataFrame.

    Parameters:
        row (Series): A row in the DataFrame.

    Returns:
        str: A trigram.
    """
    if row["Type"] == "Trigram":
        return row["Word1"] + "+" + row["Word2"] + "+" + row["Word3"]
    else:
        return ""

def filter_to_british_airways(data):
    """
    Filters the database to British Airways reviews.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: Filtered DataFrame containing only British Airways reviews.
    """
    data.dropna(subset=['Review'], inplace=True)
    data = data[data['AirlineName'] == "british-airways"]
    return data

def preprocess_tokenize_reviews(data):
    """
    Preprocesses and tokenizes the reviews in the DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with preprocessed and tokenized reviews.
    """
    data["preprocessed_review"] = data.Review.map(preprocess_text)
    data['TokenizedReview'] = data['preprocessed_review'].apply(lambda x: filter_tokens(TextBlob(x).words))
    return data

def generate_ngrams(tokenized_review, n):
    """
    Generates n-grams from tokenized words.

    Parameters:
        tokenized_review (list): A list of tokenized words.
        n (int): The size of n-grams to generate.

    Returns:
        list: A list of n-grams.
    """
    return list(zip(*[tokenized_review[i:] for i in range(n)]))

def add_ngrams_columns(data):
    """
    Adds columns for bigrams and trigrams to the data.

    Parameters:
        data (DataFrame): The data containing the 'TokenizedReview' column.

    Returns:
        DataFrame: The data with additional columns for bigrams and trigrams.
    """
    # Generate bigrams and trigrams from tokenized words
    data['Bigrams'] = data['TokenizedReview'].apply(lambda x: generate_ngrams(x, 2))
    data['Trigrams'] = data['TokenizedReview'].apply(lambda x: generate_ngrams(x, 3))
    
    return data

def calculate_sentiment_scores(data):
    """
    Calculates sentiment scores for each word, bigram, and trigram in the data.

    Parameters:
        data (DataFrame): The data containing the columns 'TokenizedReview', 'Bigrams', and 'Trigrams'.

    Returns:
        DataFrame: The data with additional columns for word, bigram, and trigram sentiment scores.
    """
    # Calculate sentiment scores for each word
    data['WordSentiment'] = data['TokenizedReview'].apply(lambda x: [TextBlob(word).sentiment.polarity for word in x])
    
    # Calculate sentiment scores for each bigram
    data['BigramSentiment'] = data['Bigrams'].apply(lambda x: [TextBlob(' '.join(bigram)).sentiment.polarity for bigram in x])
    
    # Calculate sentiment scores for each trigram
    data['TrigramSentiment'] = data['Trigrams'].apply(lambda x: [TextBlob(' '.join(trigram)).sentiment.polarity for trigram in x])
    
    return data

def explode_words(data):
    """
    Explodes words in the DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with exploded words.
    """
    word_df = data.explode(['TokenizedReview','WordSentiment'])
    word_df['Word1'] = word_df['TokenizedReview']
    word_df['Word2'] = ""
    word_df['Word3'] = ""
    word_df['Sentiment'] = word_df['WordSentiment']
    word_df['Type'] = 'Word'
    return word_df

def explode_bigrams(data):
    """
    Explodes bigrams in the DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with exploded bigrams.
    """
    bigram_df = data.explode(['Bigrams','BigramSentiment'])
    bigram_df['Bigrams'] = bigram_df['Bigrams'].apply(sorted)
    bigram_df['Word1'] = bigram_df['Bigrams'].apply(lambda x: x[0])
    bigram_df['Word2'] = bigram_df['Bigrams'].apply(lambda x: x[1])
    bigram_df['Word3'] = ""
    bigram_df['Sentiment'] = bigram_df['BigramSentiment']
    bigram_df['Type'] = 'Bigram'
    return bigram_df

def explode_trigrams(data):
    """
    Explodes trigrams in the DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with exploded trigrams.
    """
    trigram_df = data.explode(['Trigrams','TrigramSentiment'])
    trigram_df['Trigrams'] = trigram_df['Trigrams'].apply(sorted)
    trigram_df['Word1'] = trigram_df['Trigrams'].apply(lambda x: x[0])
    trigram_df['Word2'] = trigram_df['Trigrams'].apply(lambda x: x[1])
    trigram_df['Word3'] = trigram_df['Trigrams'].apply(lambda x: x[2])
    trigram_df['Sentiment'] = trigram_df['TrigramSentiment']
    trigram_df['Type'] = 'Trigram'
    return trigram_df

def join_exploded_dataframes(word_df, bigram_df, trigram_df):
    """
    Joins the exploded DataFrames into one.

    Parameters:
        word_df (DataFrame): The DataFrame with exploded words.
        bigram_df (DataFrame): The DataFrame with exploded bigrams.
        trigram_df (DataFrame): The DataFrame with exploded trigrams.

    Returns:
        DataFrame: The concatenated DataFrame.
    """
    return pd.concat([word_df, bigram_df, trigram_df])


if __name__ == '__main__':
    # Load the data from the CSV file
    data = etl.load_data('data/AirlineReviews_cleaned.csv')
    # Get the size of the loaded data
    etl.get_size(data)
    # Filter the dataset to include only British Airways reviews
    dataset_filtered = filter_to_british_airways(data)
    # Preprocess and tokenize the reviews in the filtered dataset
    dataset_tokenized = preprocess_tokenize_reviews(dataset_filtered)
    # Add columns for bigrams and trigrams to the tokenized dataset
    dataset_ngrams_added = add_ngrams_columns(dataset_tokenized)
    # Calculate sentiment scores for words, bigrams, and trigrams in the dataset
    dataset_sentiment_scores_added = calculate_sentiment_scores(dataset_ngrams_added)
    # Explode words, bigrams, and trigrams into separate rows
    word_df = explode_words(dataset_tokenized)
    bigram_df = explode_bigrams(dataset_tokenized)
    trigram_df = explode_trigrams(dataset_tokenized)
    # Join the exploded DataFrames into one
    result_df = join_exploded_dataframes(word_df, bigram_df, trigram_df)
    # Apply functions to create "Bigram" and "Trigram" columns
    result_df["Bigram"] = result_df.apply(create_bigram, axis=1)
    result_df["Trigram"] = result_df.apply(create_trigram, axis=1)
    # Select required columns
    result_df = result_df[['row_id', 'Word1','Word2','Word3','Sentiment','Type', 'Bigram', 'Trigram']]
    # Save the processed DataFrame to a CSV file
    result_df.to_csv('data/BA_sentiments.csv', index=False)
    # Print a success message
    print("Script ran successfully.")