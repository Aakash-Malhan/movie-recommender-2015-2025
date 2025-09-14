Movie Recommender (2015–2025) — MovieLens-25M + Matrix Factorization

Live Demo : https://huggingface.co/spaces/aakash-malhan/aakash-malhan_movie-recs-2015-2025

Movie recommender system built on MovieLens-25M (only movies from 2015–2025) using Matrix Factorization (SVD).
Includes an interactive Gradio app deployed on Hugging Face Spaces.

What it does

Pick a few movies you like (2015–2025).
Get top-K personalized recommendations instantly.
Runs free on CPU with permanent hosting.

Tech Stack
Python | NumPy | Pandas | SciPy | scikit-learn (TruncatedSVD)
Gradio for the web app
MovieLens-25M dataset (filtered to 2015–2025 movies)
