# Forecasting Realized Volatility Using Supervised Learning

An out-of-sample evalution to compare the accuracy of forecasted realized volatility between parametric models and various machine-learning methods.  

The files of interest in this repository are:

1. forecasting-realized-volatility.pdf: Rendered report. For improved reading experience, most code chunks are not displayed in this version
2. forecasting-realized-volatility.Rmd: Complete report including all fully-reproducible R code chunks
3. references.bib: List of references used for rendering the *.Rmd file
4. forecasting-realized-volatility.r: R script to reproduce the main results in the report
5. data/EURUSD_realized_volatility.RData: Dataset with `training` and `validation` partitions
6. data/EURUSD_quotes.csv: External, daily EUR/USD quotes used for GARCH modeling in the appendix of the report

Note that all code is designed for R version 3.6.1 for Microsoft Windows.
