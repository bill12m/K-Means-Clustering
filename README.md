# K-Means-Clustering
Project 4 from my graduate data mining course. Instructions included below. Find the final submission in Project_4.ipynb.

The undergraduate section will do Part A only; the graduate section will do both the parts. 

A.[For both G and UG] Implement the k-means clustering algorithm in a language of your choice (e.g., Python, C, C++, Java) (do not use 
any software package where this is available as a ready-to-use component). Consider only two dimensions. Create your own data (do not 
copy from any repository; do not use ready-made blobs; do not bias data to create circular clusters). Use 50 uniformly randomly generated 
two-dimensional real-valued data points in the square 1.0 <= x, y <= 100.0.  Show results separately for two differentcases: k = 2 and 
k = 4. Usethe standard distance metric(squared Euclidean).Briefly describe your stopping condition (did you achieve convergence? how many 
iterations?).For each iteration of the algorithm, for each cluster, print in a tabular fashion (andin an optionalplotif you can; no 
penalties or extra credits for including/excluding plots) the values of ∑(||𝑥−𝑚𝑒𝑎𝑛𝑗||2)2𝑥∈𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑗(intra-cluster squared-distances from the 
mean),∑||𝑥−𝑚𝑒𝑎𝑛𝑗||2𝑥∈𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑗(intra-cluster distances from the mean), and the corresponding totals for all the clustersat that iteration. 
That is, the table should like:
Iteration#    Clust#1                   Clust#2         Clust#3                     Clus#4                        TSSE(sq-dist) TSE(dist)
      Intra-sq-dist Intra-dist  Intra-sq-dist  Intra-dist  Intra-sq-dist   Intra-dist    Intra-sq-dist  Intra-dist
      1  200.25     12.3          100.64      15.8            300.4           19.7            36.0          5.2 total cols 2,4,6,8 total cols 3,5,7,9
      2  164.0     10.9           etc........... 
      
Note the difference between TSSE (total sum of squared errors; “error” = distance) and TSE (total sum of errors) in the table heading 
above: the former involves squaring, the latter does not. Note also that you will always run the algorithm to minimize the standard k-means
objective function (which involves squared-Euclidean-distances; review my notes and recordings), but for producing the results in the 
above table, you will have to calculate the sum of (plain, non-squared) distancesin addition to the sum of squared distances.Do the above 
for k=2 and k=4, separately.Note that the entire solution to the problem at any iteration is a cluster-configuration (or a collection of 
individual clusters) that consists of 2 individual clusters for k=2 (and 4 individual clusters for k=4).  Does your output show any 
iterationwhere the TSSE (sq-dist) values for the entire solution or the intra-cluster-sq-dist values (for any individual cluster)increase
with iterations(even for a single iteration)? (An easy way to check this is to visually inspect each column in the above table and see 
if the values in a given column are non-increasing.) If yes, point out the iteration number(s)where this behavior is observedand 
highlight the corresponding row(s) in the output table. If not, say so explicitly. Also, answer the same question (on possible increase
with iterations) for the non-squared scenario, that is, for intra-cluster-dist values (for any individual cluster) and TSE (dist) values
for the entire solution.Answer this question separately for the two cases k=2 and k=4.Re-run your algorithm (but do not produce any 
further printed output for submission) for at least 10 or more (the larger the better;depends on your computer; no extra credit for 
higher numbers, though) different, independent(by varying the random seed) sets of 50 input points, and state  in how manyof those 10 
(or more) independent repetitions (runs) of the experiment, you observed any iteration where the TSSE(sq-dist) or the TSE(dist) 
increased. Answer this question separately for the two cases k=2 and k=4.Avery simple implementation ofthe basic k-means is needed; 
do notuseany other version (such as k-means++ or bisecting k-means). Submit the source code, input data and clustering results (for 
any single run) as a single Word or pdf file. If you cannot manage to put everything together in a single pdf, let me know in advance.
Make sure that your results are reproducible –do not use any system-generated values for your initial (or other) seed(s) for the random 
number generator. Store all your seeds etc. in a file (or as part of code) so that any single run can be replicated at will. 

B.[For G only] Re-do the same problemwith k-medoids (for k= 4 only). Use this definition of the medoid (review my notes and recording): 
𝑐∗= argm𝑖𝑛𝑐∈{𝑥1,...,𝑥𝑛}∑‖𝑥𝑗−𝑐‖22𝑛𝑗=1. But produce an output table exactly as in part A–note that this table has nothing to do with medoids.
That is, we run k-medoids and yet observe the usual TSSE and TSE values with respect to the mean (not medoid).Answer all the questions 
as in part A. Avoid the part on repetition for 10 or more runs; a single run will do.
