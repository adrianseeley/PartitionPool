+ in extreme random trees grown to completion we are orthogonally partitioning the space down to each data point, though only on a subset of data points, and on a subset of features
+ literally 'if i look at it from this angle, what does it most look like'
+ we combine these insights and the majority of impressions tells us what the data point most likley is
+ if we take a subset of points and features, and create a small knn k=1, we are effectively creating an extreme random tree with non-orthogonal paritions
+ these k pools can be combined in the same way as extreme random trees into an extreme random forest but with non orthogonal paritions
+ this scales extremely well to gpu processing
+ an improvement may be that each pool is class balanced for labelling tasks as to prevent nearly homogenous pools (everything looks like a 9 if all you have seen are 9's)
+ a sort of out of bag complexity reduction can also occur by creating a kpool set, and attemping to remove on pool at a time agains the out of bag validation data
+ with the reduction goal being either to increase accuracy, or reduce model complexity for the same accuracy
+ the world is extremely fuzzy, so we must create models that are the exact right amount of fuzzy to match the world
+ this methodology could be extended to image recognition via

1. create NxN patches of each training image, producing a set of (patch, label) pairs
2. create a set of k pools about these patches to produce a vote of labels
3. for each training image, run all patches through the kpools to produce a set of labels equal to the number of patches
4. create a secondary set of k pools about these patch labels to produce a final label

+ many such patches would be considered useless (white space looks like all digits equally), and so the previously mentioned complexity reduction and class balancing would likely be useful