# Regional Rhythms: Spotify Audio Feature Analysis
Exploratory data analysis on the Spotify Audio features, leveraging data from the Spotify Charts Dataset and Spotify API.

- Project Report Link: [Regional Rhythms](https://docs.google.com/presentation/d/1b3KoCJx0uqqGDglQFD5NmYRAvNCMSOsmYpgOXCDUPgo/edit#slide=id.g19f02095231_2_98)
- Dataset links
  - [Spotify-API-AudioFeatures](https://drive.google.com/file/d/1pV3qGu01t87YfwytPc7yR7lXROiKYL8t/view?usp=sharing)
  - [Charts](https://drive.google.com/file/d/1AiTYbA8ZZK5A3xydtF4VigT49VgOVbHi/view?usp=sharing)

---
## Installation
After installing the repository, please adjust the folder structure as follows by moving dataset files into the `dataset` folder.:
```
Data Literacy Project
├── data
│   ├── audio_features.csv
│   ├── charts_preprocessed.csv
|   └─────────────────
├── env
├── exp
│   ├── data_audio_features.ipynb
│   ├── data_exploration.ipynb
│   ├── data_feature_vectors.ipynb
│   ├── data_rank_analysis.ipynb
│   ├── data_regional_correlation.ipynb
│   ├── data_regional_correlation_plots.ipynb
│   ├── data_regional_correlations.ipynb
│   ├── data_visualization.ipynb
│   └─────────────────
├── figures
├── output
├── source
│   ├── utils
│   ├── .env
│   ├── build_dataset.py
│   └─────────────────
├── .gitignore
├── requirements.txt
├── README.md
└─────────────────
```

If you want to run the scripts to collect data from Spotify API, please create a `.env` file in the ```source``` directory and add your Spotify API key.


## About
Our work explores the dynamic landscape of global music preferences using data from the [Spotify Charts Dataset](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) and [Spotify API](https://developer.spotify.com/documentation/web-api). By examining the top charting songs across various regions over a specified period, we aim to uncover patterns and correlations between regional popularity and the intrinsic features of the music, such as but not limited to danceability, energy, and loudness. The dataset includes key information such as dates, regions, and rankings, providing a comprehensive view of the shifting trends in music consumption worldwide. 
Our analysis is expected to provide valuable insights for artists, record labels, and marketers, helping them tailor their approaches based on regional preferences and the distinct audio features of popular music. Additionally, this study contributes to the academic understanding of cultural diversity in music, anticipating a wide variety of musical tastes and unique characteristics across different regions.

---

Collaborators:
- [Oğuz Ata Çal](https://github.com/OguzAtaCal)
- [Bora Kargı](https://github.com/Neroxn)
- [Karahan Sarıtaş](https://github.com/KarahanS)
- [Kıvanç Tezören](https://github.com/kivanctezoren)

---
This project is a component of the *ML-4102 Data Literacy* course, instructed by Philipp Henning in the winter of 2023.
> This course provides students with essential concepts and tools for working with large datasets, covering practical experiments and examples to discuss common pitfalls and best practices. It addresses basic statistical notions, bias, testing, and experimental design while employing foundational methods of machine learning and statistical data analysis.
