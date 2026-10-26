import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="CineSense Movie Explorer", page_icon="🎬", layout="wide")
GENRES = {
"Action":"fight battle chase explosion weapon mission combat hero",
"Adventure":"journey quest explore expedition treasure island discover",
"Comedy":"funny hilarious jokes awkward roommates laugh wedding",
"Crime":"murder criminal detective gang robbery heist police mafia",
"Drama":"family struggle relationship loss ambition emotional life",
"Fantasy":"magic dragon wizard kingdom spell mythical enchanted",
"Horror":"haunted ghost demon terrifying supernatural curse nightmare",
"Mystery":"clue secret mysterious unsolved disappearance puzzle suspect",
"Romance":"love romance romantic couple heartbreak relationship lover",
"Science Fiction":"space spaceship alien robot futuristic galaxy planet astronaut",
"Thriller":"suspense danger conspiracy kidnap hostage threat tension",
"War":"war soldier army battlefield military combat troops",
"Animation":"animated cartoon animation talking animals",
"Family":"family children parents kids siblings",
"History":"historical ancient king queen empire century",
"Music":"music musician singer band concert song composer",
"Documentary":"documentary interviews real events true story"
}
EXAMPLES = {
"Space survival": ("A brave astronaut travels to a distant planet after a disaster threatens humanity. An alien life-form attacks the research station.", ["Science Fiction","Action","Thriller"]),
"Detective mystery": ("A detective investigates mysterious murders. Each suspect hides a secret, and a clue reveals a dangerous conspiracy.", ["Mystery","Crime","Thriller"]),
"Romantic comedy": ("Two rival chefs share a kitchen for a cooking contest. Their arguments turn into friendship and an unexpected romance.", ["Romance","Comedy"]),
"Fantasy quest": ("A young orphan discovers a magical sword and joins warriors to defeat an ancient dragon and save a kingdom.", ["Fantasy","Adventure","Action"]),
"War drama": ("During a devastating war, a young soldier struggles to protect friends while facing the emotional cost of battle.", ["War","Drama"])
}
@st.cache_resource
def model_setup():
    texts, labels = [], []
    for genre, words in GENRES.items():
        terms=words.split()
        for i in range(5):
            texts.append(f"A {genre.lower()} story featuring {' '.join(terms[i:i+5])}. Characters face {' '.join(terms[:4])}.")
            labels.append([genre])
    for plot, labs in EXAMPLES.values():
        texts.append(plot); labels.append(labs)
    enc=MultiLabelBinarizer()
    y=enc.fit_transform(labels)
    model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),("nb",OneVsRestClassifier(MultinomialNB(alpha=.3)))])
    model.fit(texts,y)
    return model,enc
def lookup(title,key):
    r=requests.get("https://www.omdbapi.com/",params={"apikey":key,"t":title,"plot":"full"},timeout=20)
    r.raise_for_status()
    return r.json()
def predict(plot,model,enc):
    scores=model.predict_proba([plot])[0]
    return sorted(zip(enc.classes_,scores),key=lambda x:x[1],reverse=True)[:5]

st.markdown("""<div style="padding:25px;border-radius:18px;background:linear-gradient(120deg,#312e81,#7c2d12);color:white"><h1>🎬 CineSense Movie Explorer</h1><p>Search a movie by name to discover its storyline, genre, rating, cast and more.</p></div>""",unsafe_allow_html=True)
with st.sidebar:
    st.header("Movie database")
    key=st.text_input("OMDb API key",type="password",help="Get a free key at omdbapi.com/apikey.aspx")
    st.markdown("[Get a free OMDb API key](https://www.omdbapi.com/apikey.aspx)")
    st.caption("The key is used for requests from this session.")
model,enc=model_setup()
tab1,tab2=st.tabs(["🔎 Search by movie name","🧠 Classify a plot"])
with tab1:
    st.subheader("Movie lookup")
    with st.form("search"):
        title=st.text_input("Movie title",placeholder="Interstellar, Inception, 3 Idiots...")
        go=st.form_submit_button("Search movie",type="primary")
    if go:
        if not title.strip(): st.warning("Enter a movie title.")
        elif not key.strip(): st.error("Add your OMDb API key in the sidebar first.")
        else:
            try:
                with st.spinner("Fetching movie details..."): data=lookup(title.strip(),key.strip())
                if data.get("Response")!="True": st.error(data.get("Error","Movie not found. Try adding the year."))
                else: st.session_state["movie_data"]=data
            except Exception as e: st.error(f"Could not retrieve data: {e}")
    data=st.session_state.get("movie_data")
    if data:
        a,b=st.columns([1,2])
        with a:
            if data.get("Poster") not in (None,"N/A"): st.image(data["Poster"],use_container_width=True)
            else: st.info("Poster unavailable.")
        with b:
            st.title(f'{data.get("Title","Movie")} ({data.get("Year","")})')
            x,y,z=st.columns(3)
            x.metric("IMDb rating",data.get("imdbRating","N/A"))
            y.metric("Metascore",data.get("Metascore","N/A"))
            z.metric("IMDb votes",data.get("imdbVotes","N/A"))
            st.markdown("**Listed genre:** "+data.get("Genre","N/A"))
            st.markdown("**Director:** "+data.get("Director","N/A"))
            st.markdown("**Cast:** "+data.get("Actors","N/A"))
            st.markdown("**Runtime:** "+data.get("Runtime","N/A"))
            st.markdown("**Language / Country:** "+data.get("Language","N/A")+" / "+data.get("Country","N/A"))
        st.subheader("Storyline")
        st.write(data.get("Plot","No plot summary available."))
        plot=data.get("Plot","")
        if plot and plot!="N/A":
            st.subheader("NLP model predictions from storyline")
            results=predict(plot,model,enc)
            cols=st.columns(5)
            for i,(g,s) in enumerate(results): cols[i].metric(g,f"{s:.0%}")
            st.caption("These are demo model scores, not calibrated probabilities. The listed genres above come from the movie database.")
with tab2:
    sample=st.selectbox("Example plot",["Choose example..."]+list(EXAMPLES))
    default=EXAMPLES.get(sample,("",[]))[0]
    plot=st.text_area("Movie storyline",value=default,height=160,placeholder="Paste a plot summary here...")
    if st.button("Predict genres",type="primary"):
        if len(plot.strip())<20: st.warning("Enter a longer plot summary.")
        else:
            results=predict(plot,model,enc)
            st.dataframe(pd.DataFrame(results,columns=["Predicted genre","Model score"]).assign(**{"Model score":lambda d:d["Model score"].map(lambda x:f"{x:.1%}")}),hide_index=True,use_container_width=True)
            st.bar_chart(pd.DataFrame(results,columns=["Genre","Model score"]).set_index("Genre"))
st.divider()
st.caption("For live movie details, an internet connection and free OMDb API key are required. The included NLP classifier uses a small synthetic demo dataset; train on the Kaggle dataset for valid academic evaluation.")
