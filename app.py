import streamlit as st
import pandas as pd
import pickle
import requests
import time
import numpy as np

st.set_page_config(page_title="Anime Recommendation System", layout="wide", page_icon="🎬")

# --- 1. Fetching the posters and synopsis from JikanAPI ---
@st.cache_data(show_spinner=False)
def fetch_anime_details(anime_title):
    """Fetching poster images and synopsis from the Jikan API."""
    url = f"https://api.jikan.moe/v4/anime?q={anime_title}&limit=1"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                poster_url = data['data'][0]['images']['jpg']['large_image_url']
                synopsis = data['data'][0]['synopsis']
                
                # Truncate synopsis so it fits cleanly on the card
                #if synopsis and len(synopsis) > 300:
                #    synopsis = synopsis[:300] + "..."
                    
                return poster_url, synopsis
                
        # Fallback if the API can't find the exact title
        return "https://via.placeholder.com/225x318?text=No+Image", "Synopsis not available."
        
    except Exception as e:
        return "https://via.placeholder.com/225x318?text=Error", "Error fetching API data."

# --- 2. Loading the Saved Models ---
@st.cache_resource  
def load_data():
    anime_df = pickle.load(open('data/anime_data.pkl', 'rb'))
    similarity_matrix = pickle.load(open('data/similarity_matrix_lite.pkl', 'rb'))
    svd_model = pickle.load(open('data/svd_model_final_v4.pkl', 'rb'))
    return anime_df, similarity_matrix, svd_model

anime_df, similarity_matrix, svd_model = load_data()

# --- 3. The Hybrid Recommendation Engine ---
def hybrid_recommendation(user_id, anime_title):
    # Step A: Content-Based Filtering
    anime_index = anime_df[anime_df['name'] == anime_title].index[0]
    sim_scores = similarity_matrix[anime_index]
    
    # This pulls the Top 50 candidates from your lite matrix
    anime_indices = [i[0] for i in sim_scores]
    
# Step B: Collaborative Filtering (SVD)
    predictions = []
    for i in anime_indices:
        # 1. Pull the raw ID and force it to a standard Python int
        raw_anime_id = int(anime_df.iloc[i]['anime_id'])
        anime_name = anime_df.iloc[i]['name']
        
        # 2. Force the user_id to a standard Python int
        pure_user_id = int(user_id)
        
        # 3. Predict (The SVD model will now find the match in its [1, 2, 3] dictionary)
        pred = svd_model.predict(uid=pure_user_id, iid=raw_anime_id)
        score = pred.est
        
        # 4. Display for testing (You can remove the Score text once it works!)
        display_name = f"{anime_name} (Score: {score:.2f})"
        predictions.append((display_name, score))
    # Step C: Sorting and returning Top 5
    # We sort by the predicted rating (index 1) in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return [name for name, rating in predictions[:5]]

# --- 4. UI Styling ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             /* BACKGROUND IMAGE */
             background-image: url("https://images2.alphacoders.com/100/thumb-1920-1006672.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         
         /* GLASS BOX FOR THE TITLE */
         h1 {{
             background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent box */
             padding: 20px;
             border-radius: 15px;
             border: 2px solid rgba(255, 255, 255, 0.2);
             text-align: center;
             color: white !important;
             text-shadow: 2px 2px 4px #000000;
             margin-bottom: 20px;
         }}
         
         /* MAKING ALL OTHER TEXT POP */
         .stMarkdown, p, label, .stSelectbox, .stNumberInput {{
             color: white !important;
             text-shadow: 2px 2px 5px black; /* Strong black shadow for readability */
             font-weight: bold;
             font-size: 18px;
         }}
         
         /* STYLING THE CARDS  */
         .glass-card {{
             background-color: rgba(0, 0, 0, 0.7);
             padding: 15px;
             border-radius: 10px;
             border: 1px solid rgba(255, 255, 255, 0.2);
             text-align: center;
             margin-bottom: 10px;
             height: 150px;
             display: flex;
             flex-direction: column;
             justify-content: center;
             align-items: center;
             transition: transform 0.2s;
         }}
         
         .glass-card:hover {{
             transform: scale(1.05); /* Zoom effect on hover */
             border-color: #FFD700; /* Gold border */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()

# --- 5. The App Layout ---
st.title("Anime Recommendation System")
st.write("Version 2.0 Active!")

st.sidebar.header("User Options")
user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=10000, value=1 ,step=1)
# --- DIAGNOSTIC X-RAY ---
st.warning(f"Total users memorized in cloud model: {svd_model.trainset.n_users}")
st.warning(f"Does the model recognize User {user_id}? : {user_id in svd_model.trainset._raw2inner_id_users}")
st.warning(f"Data type being passed: {type(user_id)}")
# --- THE FINAL SKELETON KEY TEST ---
sample_keys = list(svd_model.trainset._raw2inner_id_users.keys())[:3]
st.error(f"Sample User IDs in Model: {sample_keys}")
st.error(f"Sample Key Type: {type(sample_keys[0])}")
# ------------------------
selected_anime = st.selectbox("Select an Anime you have watched:", anime_df['name'].values)


if st.button("Recommend Anime"):
    with st.spinner("Calculating hybrid scores and fetching posters from MyAnimeList..."):
        # Getting the 5 titles from engine
        recommendations = hybrid_recommendation(int(user_id), selected_anime)
        
        st.markdown(f"### The Top 5 Picks for User {user_id}:")
        
        # Creating 5 columns for the cards
        cols = st.columns(5)
        
        for i, anime in enumerate(recommendations):
            with cols[i]:
                poster, synopsis = fetch_anime_details(anime)
                
                if not isinstance(synopsis, str):
                    synopsis = "Synopsis not available."

                with st.container(border=True):
                    # --- THE ZERO-NETWORK FALLBACK ---
                    # Only attempt to draw the image if we got a valid link
                    if isinstance(poster, str) and poster.startswith("http"):
                        try:
                            st.image(poster, use_container_width=True)
                        except Exception:
                            # IF MAL blocks it, do NOT use another URL. 
                            # Draw a native Streamlit info box instead.
                            st.info("🖼️ Image Blocked by API")
                    else:
                        # If the API returned nothing
                        st.info("🖼️ No Image Available")
                    # ---------------------------------
                    
                    st.markdown(f"**#{i+1} {anime}**")
                    
                    with st.expander("Read Synopsis"):
                        st.caption(synopsis)
                
                time.sleep(0.5)