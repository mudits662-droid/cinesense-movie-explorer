CineSense Movie Explorer
1. Extract this ZIP.
2. Open Command Prompt in the extracted folder.
3. Run: python -m pip install -r requirements.txt
4. Run: python -m streamlit run app.py
5. Open the URL printed, normally http://localhost:8501.

To look up real movie titles, get a free OMDb API key at https://www.omdbapi.com/apikey.aspx and paste it into the app's sidebar. Then search a title to see listed genres, storyline, IMDb rating, poster, director, cast, runtime, language and country.

The app also runs an NLP genre classifier on the plot. Its included training data is only a small demo; for final academic metrics, train using the real Kaggle TMDB dataset.
