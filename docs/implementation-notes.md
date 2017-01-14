# Implementation Notes

## Mathematics

Two considerations were taken into account when the mathematical operations were implemented:

- At the time of the implementation, the version of TensorFlow used (r0.11) lacks a lot regarding slicing and assigning values to slices.
- A vectorized implementation is generally better than an Implementation with a python for loop (usually for the possible parallelism and the fact that python for loops create a copy of the same subgraph, one for each iteration).

Most of the operations described in the paper lend can be straightforwardly implemented in TensorFlow, except possibly for two operations: the allocation weighting calculations, and the link matrix updates; as they both are described in a slicing and looping manner, which can make their current implementation look a little convoluted. The following attempts to clarify how these operations were implemented.

### Allocation Weighting

In the paper, the allocation weightings are calculated using the formula:

![](https://latex.codecogs.com/gif.latex?a_t%5B%5Cphi_t%5Bj%5D%5D%20%3D%20%281%20-%20u_t%5B%5Cphi_t%5Bj%5D%5D%29%5Cprod_%7Bi%3D1%7D%5E%7Bj-1%7Du_t%5B%5Cphi_t%5Bi%5D%5D)

This operation can be vectorized by instead computing the following formula:

![](https://latex.codecogs.com/gif.latex?%5Chat%7Ba%7D_t%20%3D%20%5Cleft%28%201%20-%20%5Chat%7Bu%7D_t%20%5Cright%29%5CPi_t%5E%5Chat%7Bu%7D)

Where ![](https://latex.codecogs.com/gif.latex?%5Chat%7Bu%7D_t) is the sorted usage vector and ![](https://latex.codecogs.com/gif.latex?%5CPi%5E%7B%5Chat%7Bu%7D%7D_t) is the cumulative product vector of the sorted usage, computed with `tf.cumprod`. With this equation, we get the allocation weighting ![](https://latex.codecogs.com/gif.latex?%5Chat%7Ba%7D_t) out of the original order of the memory locations. We can reorder it into the original order of the memory locations using `TensorArray`'s scatter operation using the free-list as the scatter indices.

```python
shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

mapped_free_list = free_list + self.index_mapper
flat_unordered_allocation_weighting = tf.reshape(unordered_allocation_weighting, (-1,))
flat_mapped_free_list = tf.reshape(mapped_free_list, (-1,))
flat_container = tf.TensorArray(tf.float32, self.batch_size * self.words_num)

flat_ordered_weightings = flat_container.scatter(
    flat_mapped_free_list,
    flat_unordered_allocation_weighting
)

packed_wightings = flat_ordered_weightings.pack()
return tf.reshape(packed_wightings, (self.batch_size, self.words_num))
```

Because `TensorArray` operations work only on one dimension and our allocation weightings are of shape *batch_size × N*, we map the free-list indices to their values as if they point to consecutive locations in a flat container. Then we flat all the operands and reshape them back to their original 2D shapes at the end. This process is depicted in the following figure.

![](../assets/allocation_weighting.png)

### Link Matrix

 The paper's original formulation of the link matrix update is and index-by-index operation:

 ![](https://latex.codecogs.com/gif.latex?L_t%5Bi%2Cj%5D%20%3D%20%281%20-%20%5Cmathbf%7Bw%7D%5E%7Bw%7D_%7Bt%7D%5Bi%5D%20-%20%5Cmathbf%7Bw%7D%5E%7Bw%7D_%7Bt%7D%5Bj%5D%29L_%7Bt-1%7D%5Bi%2Cj%5D%20&plus;%20%5Cmathbf%7Bw%7D%5E%7Bw%7D_%7Bt%7D%5Bi%5D%5Cmathbf%7Bp%7D_%7Bt-1%7D%5Bj%5D)

A vectorized implementation of this operation can be written as:

![](https://latex.codecogs.com/gif.latex?L_t%20%3D%20%5B%281%20-%20%28%5Cmathbf%7Bw%7D_t%5E%7Bw%7D%5Coplus%20%5Cmathbf%7Bw%7D_t%5E%7Bw%7D%29%29%5Ccirc%20L_%7Bt-1%7D%20&plus;%20%5Cmathbf%7Bw%7D_t%5E%7Bw%7D%5Cmathbf%7Bp%7D_%7Bt-1%7D%5D%5Ccirc%20%281-I%29)

Where ![](https://latex.codecogs.com/gif.latex?%5Ccirc) is elementwise multiplication, and ![](https://latex.codecogs.com/gif.latex?%5Coplus) is a *pairwise addition* operator defined as:

![](https://latex.codecogs.com/gif.latex?u%20%5Coplus%20v%20%3D%20%5Cbegin%7Bpmatrix%7D%20u_1%20&plus;%20v_1%20%26%20%5Chdots%20%26%20u_1&plus;v_n%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%5C%5C%20u_n&plus;v_1%20%26%20%5Chdots%20%26%20u_n&plus;v_n%20%5Cend%7Bpmatrix%7D)

Where ![](https://latex.codecogs.com/gif.latex?%5Cinline%20u%2Cv%20%5Cin%20%5Cmathbb%7BR%7D%5En). This allows TensorFlow to parallelize the operation, but of course with a cost incurred on the space complexity.

*The elementwise multiplication by ![](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cmathit%7B1%20-%20I%7D) is to ensure that all diagonal elements are zero, thus ensuring the elimination of self-links.*


## Weight Initializations

* **Memory's usage and precedence vectors and link matrix** are initialized to zero as specified by the paper.

* **Memory's matrix, read and write weightings, and read vectors** are initialized to a very small value (10⁻⁶). Attempting to initialize them to 0 resulted in **NaN** after the first few iterations.

*These initialization schemes were chosen after many experiments on the copy-task, as they've shown the highest degree of stability in training (The highest ratio of convergence, and the smallest ratio of NaN-outs). However, they might re-consideration with other tasks.*
