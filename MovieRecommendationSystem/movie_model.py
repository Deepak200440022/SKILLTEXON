import ast
class MovieInfo:
    def __init__(self, id, title, poster_path, release_date=None, runtime=None, tagline=None,
                 overview=None, adult=False, cast=None, similarity=None, genre=None, rating=None):
        self.id = id
        self.title = title or "Unknown Title"
        self.poster_path = f"https://image.tmdb.org/t/p/original/{poster_path}" if poster_path else ""
        self.release_date = release_date or "Unknown"
        self.runtime = f"{int(runtime)} minutes" if runtime not in (None, "") else "Runtime Unavailable"
        self.tagline = tagline or ""
        self.overview = overview or ""
        self.adult = adult
        self.cast = cast or []
        self.similarity = similarity
        self.genre = genre or "Unknown Genre"
        self.rating = f"{round(rating, 1)}/10 IMDb" if rating is not None else "Rating Unavailable"

    @classmethod
    def from_row(cls, row):
        genres_data = row.get("genres", [])
        genres_data = ast.literal_eval(genres_data)
        genre_names = ", ".join(genre["name"] for genre in genres_data if "name" in genre)
        cast_data = row.get("cast", "[]")
        cast_data = ast.literal_eval(cast_data)
        sorted_cast = sorted(cast_data, key=lambda x: x.get("order", float("inf")))

        return cls(
            id=row.get("id"),
            title=row.get("title"),
            poster_path=row.get("poster_path", ""),
            release_date=row.get("release_date"),
            runtime=row.get("runtime"),
            tagline=row.get("tagline"),
            overview=row.get("overview"),
            adult=row.get("adult", False),
            cast=sorted_cast,
            genre= genre_names,
            rating=row.get("vote_average"),
        )
