# Exploration of Poisson Statistics with Radioactive Decay 

As of 09/03/2026, this repository contains raw data, analysis code (Python + Jupyter) and all the generated figures for a laboratory on the exploration of Poisson Statistics through the lens of radioactive decay. Once completed, this repo will also contain a lab logbook (PDF) LaTeX source code, and compiled PDF report. 

The goal was to explore the application of Poisson statistics onto random and discrete counting events, in this case radioactive decay. Data was collected using a Geiger counter. The variables of sampling rate, distance to radioactive source, and total time window of collection were varied to see how it impacted the fit of the Poisson distribution onto the histogram of observed events.  

## Repository structure
\- `raw_data/` - raw data files in subfolders corresponding to independent variable explored

\- `notebooks/` - code/notebooks to analyze the data and generate figures 

\- `src/` -general purpose labtools helper module

\- `figures/` - figures used in the report (generated from analysis and illustrative purpose ones) Each sub-folder has a category corresponding to the raw data folder it was based off of, and each figure is named after the corresponding raw data file. 

\- `report/` - LaTeX source (Overleaf export) and lab logbook PDF (Will be available on 12/03/2026)

\- `output/photoelectric\_report.pdf` — compiled report PDF (Will be available on 12/03/2026)

## Note on the use of AI assistance
## Note on AI assistance
Portions of the code were written with AI assistance for scaffolding and refactoring. AI assistance was also called upon to accelerate the processes of debugging, documentation consultation and brainstorming. All AI outputs were reviewed, tested, and adapted as needed by me.

## Authors
Justine Thebault-Weiser, Ari Polterovich and Bogdan-Vladimir Damian (Me)
