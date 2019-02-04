# Sentiment analysis
The SAR14 IMDB movie review dataset (Download from http://tabilab.cmpe.boun.edu.tr/datasets/review_datasets/SAR14.zip) was used. The  data consisted of a detailed review and its respective rating. All the reviews had a rating, ranging
from 1 to 10. We decided to generate 3 models based on these ratings. A Binary class model, where all the reviews with the rating of 1 to 5 are label as Negative and all the
reviews with the rating of 6 to 10 are Positive. A Tertiary class model, where all the reviews with the rating of 1 to 3 are Negative, all the reviews
with the rating of 4 to 6 are Neutral and all the reviews with the rating of 7 to 10 are Positive. A Five class model, where all the reviews with the rating of 1 to 2 are Highly Negative, all the reviews
with the rating of 3 to 4 are Negative, all the reviews with the rating of 5 to 6 are Neutral, all the reviews with the rating of 7 to 8 are Positive, and all the reviews with the rating of 9 to 10 are Highly
Positive. Code for the same can be found in the GenerateDataset file. 

When running the models make sure the generated CSVs are within a "dataset" folder.

A detailed analysis has been provided in the report titled Sentiment Analysis

# License

Copyright 2019 Mir Nawab Hussain, Muhammad Aatif Ishtiaq, Muhammad Abubakar Dar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
