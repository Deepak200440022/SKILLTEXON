from flask import Flask, render_template, jsonify, request, abort
from dataset_summary import merged_df
from movie_model import MovieInfo
import ast
from recomendaton_model import recommend_similar_movies
app = Flask(__name__)


def extract_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return genres[0]["name"] if genres else "Unknown"
    except:
        return "Unknown"


@app.route('/')
def index():
    sample = merged_df[["id", "title", "poster_path", "genres", "vote_average"]].dropna(subset=["title", "poster_path"]).sample(3)
    movie_list = [MovieInfo.from_row(row) for _, row in sample.iterrows()]



    return render_template("index.html", movies=movie_list)

# Route to serve suggestions
@app.route("/suggest", methods=["GET"])
def suggest():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify([])

    matches = merged_df[merged_df["title"].str.lower().str.contains(query, na=False)].head(10)
    results = matches[["title", "poster_path", "id"]].fillna("").to_dict(orient="records")

    return jsonify(results)

@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    movie_row = merged_df[merged_df["id"] == movie_id]
    if movie_row.empty:
        abort(404)

    row = movie_row.iloc[0]
    movie = MovieInfo.from_row(row)

    try:
        recommendations = recommend_similar_movies(row["title"], top_n=8)
    except ValueError:
        recommendations = []

    rec_movies = []
    for idx, score in recommendations:
        rec_row = merged_df.iloc[idx]
        rec_movie = MovieInfo.from_row(rec_row)
        rec_movie.similarity = round(float(score), 2)
        rec_movies.append(rec_movie)

    return render_template("movie_detail.html", movie=movie, recommendations=rec_movies)


if __name__ == "__main__":
    app.run(debug=True)