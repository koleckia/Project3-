# start from the rstudio/plumber image
FROM rstudio/plumber

# install the linux libraries needed for plumber
RUN apt-get update -qq && apt-get install -y libssl-dev libcurl4-openssl-dev libpng-dev libxml2-dev libgit2-dev pandoc make g++ libglpk-dev

# install packages
RUN R -e "install.packages(c('tidyverse', 'tidymodels', 'yardstick','ggplot2','glmnet','tree','baguette','parsnip', 'ranger'))"

# copy myAPI.R from the current directory into the container
COPY plumber.R plumber.R
COPY diabetes_binary_health_indicators_BRFSS2015.csv diabetes_binary_health_indicators_BRFSS2015.csv


# open port to traffic
EXPOSE 8000

# when the container starts, start the myAPI.R script
ENTRYPOINT ["R", "-e", \
            "pr <- plumber::plumb('plumber.R'); pr$run(host='0.0.0.0', port=8000)"]
