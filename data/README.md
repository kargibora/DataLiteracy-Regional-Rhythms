 Download the required datasets:
  * [Spotify-API-AudioFeatures](https://drive.google.com/file/d/1pV3qGu01t87YfwytPc7yR7lXROiKYL8t/view?usp=sharing)
  * [Charts](https://drive.google.com/file/d/1AiTYbA8ZZK5A3xydtF4VigT49VgOVbHi/view?usp=sharing)

---

All the correlations of audio featuers between rank, streams and popularity is saved under the `correlations/` folder. Such files holds the following objects:
```yaml
COUNTRY_1:
  - PEARSON_CORRS
  - SPEARMAN_CORRS
  - KENDALL_CORRS
COUNTRY_2:
  - PEARSON_CORRS
  - SPEARMAN_CORRS
  - KENDALL_CORRS
...
```

where each correlation object is a dictionary where keys are audio features and values are the calculated correlation values.
