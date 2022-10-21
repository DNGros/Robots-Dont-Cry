This data comes from data_export.py in the source code.

There are few categories of files.

## Classifiable data

If you just want to treat data as a decision problem, 
then the classifiable_df*.csv files are probably what you want. 

It contains one row for each utterance per question/embodiment.

The key columns:
* resp_cat: An identifier that fully describes question context. For example "comfort_r-humanoid" or "truthful_r-chatbot"
* question_topic: An identifier for whether it is robot-comfort or robot-truthful (no info on humanoid or chatbot). Note we dont consider the "if instead said by a human" questions in the classifiable data.
* majority_prob: The estimated probability that answered afirmatively per our bootstrap process.
* turn_a: first turn of conversation (said by human)
* turn_b: the response by the bot
* joined_text: a combination of the two turns and the question. Can be passed into a generic text classifier.
* base_majority_vote: the origional vote ignoreing the bootstrapping process

## Joined results

The joined results join together response data with the survey text data and demographic data. There is one row for every answer for every question (so 4 questions for every utterance per participant). 

The most useful one is joined_results_filt_threshed.csv which only includes ratings passing our various quality filters. There is also joined_results_filt.csv which is similar, but allows a few utterances with <3 ratings.

There is the joined_results_raw.csv which is all collected data points.
