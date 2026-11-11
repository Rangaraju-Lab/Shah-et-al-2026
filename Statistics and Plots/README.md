# Shah et al. 2026 figure reports

Some source files, plots, and statistical results are pending a more recent update. The affected panels are marked **update for 2026 pending** in the reports.

## Open a report

On GitHub, choose **Code → Download ZIP** and extract the archive. Open the `.html` file in the relevant figure folder in your browser. The reports contain the analysis tables and plots and can be viewed offline.

## Rerun an analysis

Keep each figure’s Rmd and data files together. With R and RStudio installed, install the required R packages once.

```r
install.packages(c("rmarkdown", "knitr", "dplyr", "readr", "tidyr", "ggplot2",
                   "ggbeeswarm", "ggtext", "afex", "emmeans", "tibble", "scales"))
```

Open the figure’s `.Rmd` in RStudio and click **Knit**. This regenerates the HTML in the figure folder and saves PNGs under `Graphs/`.

For panel **Fig 1J**, keep `Fig 1J.csv.gz` in the `Fig 1` folder and knit `Fig 1.Rmd`. The script reads the compressed CSV directly.
