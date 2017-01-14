### Common Settings

Both series and single models were trained on 2-layer feedforward controller (with hidden sizes 128 and 256 respectively) with ReLU activations, and both share the following set of hyperparameters:

- RMSProp Optimizer with learning rate of 10⁻⁴, momentum of 0.9.
- Memory word size of 10, with a single read head.
- Controller weights are initialized from samples 1 standard-deviation away from a zero mean normal distribution with a variance ![](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Csigma%5E2%20%3D%20%5Ctext%7Bmin%7D%5Chspace%7B0.2em%7D%5Cleft%281%5Ctimes10%5E%7B-4%7D%2C%20%5Cfrac%7B2%7D%7Bn%7D%5Cright%29), where ![](https://latex.codecogs.com/gif.latex?%5Cinline%20n) is the size of the input vector coming into the weight matrix.
- A batch size of 1.

All output from the DNC is squashed between 0 and 1 using a sigmoid functions and  binary cross-entropy loss (or logistic loss) function of the form:

![loss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%28y%2C%20%5Chat%7By%7D%29%20%3D%20-%5Cfrac%7B1%7D%7BBTS%7D%5Csum_%7Bi%3D1%7D%5E%7BB%7D%5Csum_%7Bj%3D1%7D%5E%7BT%7D%5Csum_%7Bk%3D1%7D%5ES%5Cleft%28%20y_%7Bijk%7D%5Clog%20%5Chat%7By%7D_%7Bijk%7D%20&plus;%20%281%20-%20y_%7Bijk%7D%29%5Clog%281-%5Chat%7By%7D_%7Bijk%7D%29%20%5Cright%29)

is used. That is the mean of the logistic loss across the batch, time steps, and output size.

All gradients are clipped between -10 and 10.

*Possible __NaNs__ could occur during training!*


### Series Training

The model was first trained on a length-2 series of random binary vectors of size 6. Then starting off from the length-2 learned model, a length-4 model was trained in a **curriculum learning** fashion.

The following plots show the learning curves for the length-2 and length-4 models respectively.

![series-2](/assets/model-series-2-curve.png)

![series-4](/assets/model-series-4-curve.png)

*Attempting to train a length-4 model directly always resulted in __NaNs__. The paper mentioned using curriculum learning for the graph and mini-SHRDLU tasks, but it did not mention any thing about the copy task, so there's a possibility that this is not the most efficient method.*

#### Retraining
```
$python tasks/copy/train-series.py --length=2
```
Then, assuming that the trained model from that execution is saved under the name 'step-100000'.

```
$python tasks/copy/train-series.py --length=4 --checkpoint=step-100000 --iterations=20000
```

### Single Training

The model was trained directly on a single input of length between 1 and 10 and the length was chosen randomly at each run, so no curriculum learning was used. The following plot shows the learning curve of the single model.

![single-10](/assets/model-single-curve.png)

#### Retraining

```
$python tasks/copy/train.py --iterations=50000
```
