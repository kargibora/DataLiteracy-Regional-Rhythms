# Regional Rhythms: Spotify Audio Feature Analysis
Exploratory data analysis on the Spotify Audio features, leveraging data from the Spotify Charts Dataset and Spotify API.

[Project Report](https://docs.google.com/presentation/d/1b3KoCJx0uqqGDglQFD5NmYRAvNCMSOsmYpgOXCDUPgo/edit#slide=id.g19f02095231_2_98)

---
## About
Our work explores the dynamic landscape of global music preferences using data from the [Spotify Charts Dataset](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) and [Spotify API](https://developer.spotify.com/documentation/web-api). By examining the top charting songs across various regions over a specified period, we aim to uncover patterns and correlations between regional popularity and the intrinsic features of the music, such as but not limited to danceability, energy, and loudness. The dataset includes key information such as dates, regions, and rankings, providing a comprehensive view of the shifting trends in music consumption worldwide. 
Our analysis is expected to provide valuable insights for artists, record labels, and marketers, helping them tailor their approaches based on regional preferences and the distinct audio features of popular music. Additionally, this study contributes to the academic understanding of cultural diversity in music, anticipating a wide variety of musical tastes and unique characteristics across different regions.

### Collaborators:
- [Oğuz Ata Çal](https://github.com/OguzAtaCal)
- [Bora Kargı](https://github.com/Neroxn)
- [Karahan Sarıtaş](https://github.com/KarahanS)
- [Kıvanç Tezören](https://github.com/kivanctezoren)

---

*This project is a component of the ML-4102 Data Literacy course, instructed by Philipp Henning in the winter of 2023.*
> This course provides students with essential concepts and tools for working with large datasets, covering practical experiments and examples to discuss common pitfalls and best practices. It addresses basic statistical notions, bias, testing, and experimental design while employing foundational methods of machine learning and statistical data analysis.

## Installation

1. We have used Python 3.10 to obtain our results. Although the requirements should work with higher (and some lower) Python versions, you may want to ensure working Python 3.10 environment for reproducability.

Environment creation example with Conda:

```
$ conda create --name regional_rhythms python=3.10
$ conda activate regional_rhythms
```

2. Install pip if you haven't in order to install the requirements:

Example installation in a Conda environment:

```
$ conda install pip
```

3. Install the requirements:

```
$ pip install -r requirements.txt
```

4. Download the required datasets:
  * [Spotify-API-AudioFeatures](https://drive.google.com/file/d/1pV3qGu01t87YfwytPc7yR7lXROiKYL8t/view?usp=sharing)
  * [Charts](https://drive.google.com/file/d/1AiTYbA8ZZK5A3xydtF4VigT49VgOVbHi/view?usp=sharing)

5. If you would like to run the scripts that collects data from Spotify API, please create a `.env` file in the `source/` directory and add your Spotify API key:

```
API_KEY="Spotify API Key goes here"
```

6. Move the related dataset files into the `data/` folder, resulting in a file structure as follows:

```
Data Literacy Project
├── data
│   ├── audio_features.csv
│   ├── charts_preprocessed.csv
│   ├── ...
│   └─────────────────
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
│   ├── ...
│   └─────────────────
├── output
│   ├── ...
│   └─────────────────
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

The `.ipynb` files under `exp/` can now be executed.

## Appendix

We present our obtained plots under the `figures/` directory. The directory includes additional plots to those included in the report, along with their explanations in a [README file](./figures/README.md).

We also present our processed data in the `data/` folder as a convenience, which are explained further in the folder's [README file](./data/README.md).