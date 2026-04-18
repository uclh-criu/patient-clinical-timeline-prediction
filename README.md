# Patient Clinical Timeline Prediction

This project builds on the UCLH-CRIU clinical timeline extraction work:  
[uclh-criu/patient-clinical-timeline-extraction](https://github.com/uclh-criu/patient-clinical-timeline-extraction/tree/master), focussing on downstream prediction using the extracted patient timelines.

## Main notebooks

- `1. pre-processing.ipynb` - timeline cleaning, cohort filtering, and horizon-specific outputs
- `2. exploratory_data_analysis.ipynb` - data inspection and descriptive analysis
- `3a. bag_of_words.ipynb` - classical ML baselines on timeline-derived features
- `3b. sequence_model.ipynb` - sequence-based modeling experiments

## Notes

- Timelines generated from the upstream extraction project should be saved in a local `timelines/` folder in the repo root.
- `1. pre-processing.ipynb` uses the timeline files in `timelines/` as input.
- Horizon datasets are then written back to `timelines/` during pre-processing.
- Keep notebook configuration cells (`DATA`, `cohort`, `HORIZON`) aligned across runs.
