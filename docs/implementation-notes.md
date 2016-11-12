# Implementation Notes

## Mathematics

Two considerations were taken into account when the mathematical operations were implemented:

- At the time of the implementation, the version of TensorFlow used (r0.11) lacks a lot regarding slicing and assigning values to slices.
- A vectorized implementation is usually better than an Implementation with a loop (usually for the possible parallelism).

Most of the operations described in the paper lend can be straightforwardly implemented in TensorFlow, except possibly for two operations: the allocation weighting calculations, and the link matrix updates; as they both are described in a slicing and looping manner, which can make their current implementation look a little convoluted. The following attempts to clarify how these operations were implemented.

### Allocation Weighting

In the paper, the allocation weightings are calculated using the formula:

![](https://latex.codecogs.com/gif.latex?a_t%5B%5Cphi_t%5Bj%5D%5D%20%3D%20%281%20-%20u_t%5B%5Cphi_t%5Bj%5D%5D%29%5Cprod_%7Bi%3D1%7D%5E%7Bj-1%7Du_t%5B%5Cphi_t%5Bi%5D%5D)

which can be implemented naively with a loop with a runtime complexity of ![](https://latex.codecogs.com/gif.latex?%5Cinline%20O%28n%5E2%29). There is no way to escape the slice-assignment operations, because the reordering the the sorted free-list back into its original places is crucial. However, there is a possibility to make things a little faster.

```python
shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

allocation_weighting_batches = []
for b in range(self.batch_size):
    allocation_weighting = tf.zeros([self.words_num])
    unpacked_free_list = tf.unpack(free_list[b])
    for pos, original_indx in enumerate(unpacked_free_list):
        mask = tf.squeeze(tf.slice(self.I, [original_indx, 0], [1, -1]))
        allocation_weighting += mask * unordered_allocation_weighting[b, pos]
        allocation_weighting_batches.append(allocation_weighting)

return tf.pack(allocation_weighting_batches)
```
In this implementation, we calculate all the required products first on the sorted usage and get an unordered version of allocation weighting, then we use a loop to put back the weightings into their correct places. Because there is no differentiable way in TensorFlow to directly assign to slices of a tensor, a mask with 1 in the desired slice position and zeros else where is multiplied with the value to be assigned and added to the target tensor.

In this implementation, the loop is ![](https://latex.codecogs.com/gif.latex?%5Cinline%20O%28n%29), but allows the exploitation of the [possible parallelism](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) of the `cumprod` operation, which could take down the runtime complexity down to ![](https://latex.codecogs.com/gif.latex?%5Cinline%20O%5Cleft%28%5Cfrac%7Bn%7D%7Bp%7D%20&plus;%20%5Ctext%7Blg%7D%5Chspace%7B0.2em%7Dp%20%5Cright%20%29), where ![](https://latex.codecogs.com/gif.latex?p) is the number of parallel processors.

*I'm not sure if TensorFlow implements a parallel version of `cumprod` but the implementation can exploit it if it's there.*

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

* **Controller weights** are samples 1 standard-deviation away from a zero mean normal distribution with a variance ![](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Csigma%5E2%20%3D%20%5Ctext%7Bmin%7D%5Chspace%7B0.2em%7D%5Cleft%281%5Ctimes10%5E%7B-4%7D%2C%20%5Cfrac%7B2%7D%7Bn%7D%5Cright%29), where ![](https://latex.codecogs.com/gif.latex?%5Cinline%20n) is the size of the input vector coming into the weight matrix.

* **Memory's usage and precedence vectors and link matrix** are initialized to zero as specified by the paper.

* **Memory's matrix, read and write weightings, and read vectors** are initialized to a very small value (10⁻⁶). Attempting to initialize them to 0 resulted in **NaN** after the first few iterations.

*These initialization schemes were chosen after many experiments on the copy-task, as they've shown the highest degree of stability in training (The highest ratio of convergence, and the smallest ratio of NaN-outs). However, they might re-consideration with other tasks.*
