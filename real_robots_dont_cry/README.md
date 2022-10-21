The code for Robots-Dont-Cry

Some key scripts:

**join_results.py**: The code for merging the results from all our turk runs.
The get_joined_results() can be used to get unfiltered survey responses
as dataframe.

**explore_quality_check.py**: 
Code for filtering the results. Can use the get_filtered_results() 
to get a dataframe that has been filtered

**data_bootstrap.py**:
Used to run bootstrap sampling runs.
Can do someting `make_bootstrap_samples(get_filtered_results(), 100)` to 
get 100 simulations from the data. This can be used for calculating simulations
for use in boothstrapping.


 
