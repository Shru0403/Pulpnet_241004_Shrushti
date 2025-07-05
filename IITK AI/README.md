# Source code includes:
1) Data Extraction files for vox, ics and other iitk webpages respectively.
2) dataset.json which contains the processed dataset
3) Embeddings.py which creates embeddings for the dataset.json file
4) chunk_embeddings.pt which contains the processed embeddings of the dataset
5) streamlit_app.py which contains the actual streamlit interface

# Process to run it locally:
1) Open the streamlit_app.py file.
2) Run it via command: ```streamlit run streamlit_app.py```
3) A local host window will open in your browser showing the chatbot interface.
4) The chatbot will take 60-90 seconds to load initally. Afterwards type in your query and press enter. The Chatbot will show the answer.
