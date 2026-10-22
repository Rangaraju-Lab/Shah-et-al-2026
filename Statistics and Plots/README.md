# Statistics and Plots (Requires update for 2026)

This directory contains all statistical analysis and plotting workflows for the project.

---

## Directory structure

Each figure-wise subfolder contains three types of files:

1. **HTML exports**  
   - Standalone files exported from R.  
   - Contain the complete source dataset, the statistical tests applied, and the plotting code for each figure.  
   - Do **not** require R, RStudio, or separate data files.  
   - Cannot run code, but can be opened in any browser to review both code and outputs.

2. **R Markdown (.Rmd) files**  
   - Source files that can be opened and executed in RStudio.  
   - Reproduce the analyses and plots when provided with the appropriate data.

3. **Source data (.csv) files**  
   - Input datasets imported by the `.Rmd` scripts.  
   - Provide the raw values required for statistical tests and figure generation.

---

## Usage

- For a **quick review** of analyses and plots, open the `.html` exports directly in a web browser.  
- For **reproducible analysis**, open the `.Rmd` files in RStudio with the corresponding `.csv` source data.
