import pandas as pd
import  ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required nltk data
required_packages = ['punkt', 'stopwords', 'wordnet']
for pkg in required_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)



df_credits =  pd.read_csv("dataset/tmdb_5000_credits.csv")
df_moives = pd.read_csv("dataset/tmdb_5000_movies.csv")
df_posters = pd.read_csv("dataset/TMDB_movie_dataset_v11.csv", engine="pyarrow")



# Merge cast from df2 and poster from df3 into df1 using the 'id' column
merged_df = df_moives.merge(df_credits[['movie_id', 'cast']], left_on='id', right_on='movie_id', how='left') \
               .merge(df_posters[['id', 'poster_path']], on='id', how='left') \
               .drop(columns=['movie_id'])

droped_columns = ["budget",
                  "homepage",
                  "original_title",
                  "production_companies",
                  "production_countries",
                  "revenue",
                  "spoken_languages",
                  "status",
                  "vote_count",
                  ]

merged_df = merged_df.drop(columns=droped_columns)

def convert(text):
    return " ".join([i["name"].lower() for i in ast.literal_eval(text)])

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return " "

    tokens = [lemmatizer.lemmatize( word.lower()) for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words ]


    return " ".join(tokens)






# print(convert(merged_df.genres[0]))
merged_df["tags"]  =  merged_df["genres"].apply(convert) + ' ' + merged_df["keywords"].apply(convert) + ' ' + merged_df["cast"].apply(convert) + ' ' + merged_df["original_language"] \
                        + ' ' + merged_df["overview"].apply(clean_text) + ' ' + merged_df["tagline"].apply(clean_text) + ' ' + merged_df["title"].apply(clean_text)

if __name__ == "__main__":
    print("TMDB Movie Dataset Summary\n")
    print(f"Total Movies: {len(merged_df)}\n")

    print("Columns after preprocessing and merging:")
    print(list(merged_df.columns), "\n")

    print("Sample row:\n")
    print(merged_df.iloc[0][["title", "genres", "keywords", "cast", "overview", "tagline", "tags"]])
