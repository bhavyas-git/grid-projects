# grid-projects

This repository consists of code made to analyze and visualize women's representation in film in a clean and well formated way. The files should be viewed in conjunction with this [blog post](https://github.com/bhavyas-git/grid-projects/blob/main/Women's%20Representation%20in%20Film%20Blog.pdf) for optimal interpretation.

I acquired my dataset from this [website](https://bechdeltest.com/) (result.csv) using this [API documentation](https://bechdeltest.com/api/v1/doc)
This dataset came with imdb ids, movie titles, Beckdel ratings and year of release. For more features to use in analysis I have scraped the official imdb website using the imdb ids to get genres, imdb ratings and descriptions of each movie. The code for this web scrapping is in the scrape_bechdel.ipynb file. 

Next we do some data processing and preperation after which we have a set of interactive and static visualizations designed to analyze the relationship between IMDb ratings, movie genres, and Bechdel Test performance across a 10644 movie dataset. These can be found in the order below in the visuals_bechdel.ipynb file

Visuals designed are:

1. Bechdel Test Results Over Time (Stacked Area Chart)
Start here to see the big picture. This chart shows how the share of films passing and failing the Bechdel Test changes across decades, revealing long-term progress and periods of stagnation.

2. Bechdel Test Results by Genre (Diverging Bar Chart)
Next, shift from time to structure. This view compares pass and fail rates across genres, highlighting how different storytelling traditions create or limit space for women’s interactions.

3. Bechdel Score Distribution by Genre (100% Stacked Bars)
This figure adds nuance by breaking each genre into all four Bechdel outcomes, showing not just success or failure, but how close genres come to full representation.

4. IMDb Rating Distribution by Bechdel Outcome (Histogram + Density Curves)
Here, audience response enters the picture. This plot tests whether films with stronger representation are rated differently, revealing that inclusivity and popularity largely overlap.

5. Genre Distribution by IMDb Rating and Movie Count (Bubble Chart)
This visualization introduces scale and reach. Bubble size shows how widely each genre’s stories circulate, while position reflects average audience ratings, highlighting which genres shape cultural norms most strongly.

6. Bechdel Score Distributions by Selected Genres (Mini Bar Charts)
Now zoom in. These small multiples use balanced samples to compare how representation patterns differ within individual high and low rated genres.

7. IMDb Ratings and Bechdel Pass Rates by Genre and Decade (Heatmap)
Finally, everything comes together. This heatmap combines time, genre, and audience reception, revealing a patchwork of progress shaped by history and storytelling conventions.

All visuals are generated using Python (Pandas, Plotly, Matplotlib/Seaborn) and are fully reproducible from the provided notebooks and datasets. All the above visuals can be viewed in static png format in the bechdel test visuals folder. scraped dataset is already provided for ease of execution but can be reproduced using scrape_bechdel.ipynb. 
