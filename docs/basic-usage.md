# Project Structure and Usage

## Structure Overview

The implementation is structured into three main modules.

- **Memeory** `dnc/memeory.py`: this module implements the memory access and attention mechanisms used in the DNC architecture. This is considered an internal module which a basic user would need to work directly with.
- **BaseController** `dnc/controller.py`: this module defines an **abstract class** that represents the controller unit in the DNC architecture. The class abstracts away all the common operations between various task (like interface vector parsing, input and read vector concatenation, ... etc) and only leaves for the user two un-implemented methods that concern with defining the internal neural network.
- **DNC** `dnc/dnc.py`: this module integrates the operations of the controller unit and the memory, and it's considered the public API that the user should interact directly with. This module also abstracts away all the common operations across various tasks (like initiating the memory, looping through the time steps, memory-controller communications, ... etc) so the user is only required to construct an instance of that class using the desired parameters, and use simple API to feed data into the model and get outputs out of it.

The following pseudo-UML diagram summarizes the project structure:

![UML](/assets/UML.png)

The reasons behind such design choices stems from the inherit flexibility and generality of the DNC architecture and its ability to be adapted to various tasks and problems. This design was chosen to reflect these characteristics and allow the user to quickly set up his/her model by focusing on the specific details of the task without worrying about the rest of the architecture's operation. This is also the justification behind leaving any details about the loss, optimizers and session runs out of the implementation and into the users hand to adapt them to whatever task they desire.

## Usage

### Defining the Controller

The first step at setting up your task is to define your controller's internal neural network. This is done by extending the `BaseController` class and implementing the two methods that define your network:

- `network_vars(self): void`: in this method you should define your network variables and their initializers as an instance attributes of the class (aka `self.*`). This method will be used automatically by the `DNC` instance to create the variables upon construction. This method shouldn't return any thing.
- `network_op(self, X): Tensor`: in this method you define the operation of your network, that is the layers operations the activations, batch normalizations, ... etc. This method takes one input, which is a 2D Tensor of shape `batch_size X (input_size + read_vectors_size)` and should return a 2D Tensor of shape `batch_size X output_size`

When you define your `network_vars` method, you shouldn't worry about calculating the size of the input plus the read vectors, this value will be automatically available for you via the attribute `self.nn_input_size`. The defined batch size is also available via `self.batch_size`.

The following is an example of controller with a 1-layer feedforward neural network:

```python
import tensorflow as tf
from dnc.controller import BaseController

class FeedfrowardController(BaseController):
    def network_vars(self):
        self.W = tf.Variable(tf.truncated_normal([self.nn_input_size, 128]), name='weights')
        self.b = tf.Variable(tf.zeros([128]), name='bias')


    def network_op(self, X):
        output = tf.matmul(X, self.W) + self.b
        activations = tf.nn.relu(output)

        return activations
```

**Notice** that the network handles works with flat inputs and flat outputs, so if you're planning to do convolutions you should:
1. Flatten your data before passing it to the model.
2. Bring it back to 2D in the beginning of your `network_op`.
3. Flatten the output before returning it from the `network_op`.

#### Defining Recurrent Controllers

To define a controller with a recurrent neural network, you'll need to add a few things to your new controller class:
- Defining the state of your network inside your `network_vars` method.
- A method named `get_state` that returns a tuple `(previous_output, previous_hidden_state)` which should be read from the defined state.
- A method named `update_state` that will be used to update the values of the state across runs. This method should return a TensorFlow operation.
- Making the `network_op` method take an extra argument for the state and return alongside the output a state tuple.

You only need to address these changes and the `DNC` module would automatically recognize it and do the rest of the work.

The following is an example of a possible recurrent controller:
```python
import tensorflow as tf
from dnc.controller import BaseController

class RecurrentController(BaseController):
    def network_vars(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
        self.state = tf.Variable(tf.zeros([self.batch_size, 64]), trainable=False)
        self.output = tf.Variable(tf.zeros([self.batch_size, 64]), trainable=False)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def update_state(self, new_state):
        return tf.group(
            self.output.assign(new_state[0]),
            self.state.assign(new_state[1])
        )

    def get_state(self):
        return (self.output, self.state)
```

#### Initial Transformation Weights

By default, the transformation weights matrices ![](https://latex.codecogs.com/gif.latex?W_y%2C%20W_%5Czeta%2C%20W_r) are initialized from a zero-mean normal distribution with a standard deviation of 0.1. This initialization scheme is defined in the `initials` method of a `BaseController`:

```python
def initials(self):
    """
    sets the initial values of the controller transformation weights matrices
    this method can be overwritten to use a different initialization scheme
    """
    # defining internal weights of the controller
    self.interface_weights = tf.Variable(
        tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
        name='interface_weights'
    )
    self.nn_output_weights = tf.Variable(
        tf.random_normal([self.nn_output_size, self.output_size], stddev=0.1),
        name='nn_output_weights'
    )
    self.mem_output_weights = tf.Variable(
        tf.random_normal([self.word_size * self.read_heads, self.output_size],  stddev=0.1),
        name='mem_output_weights'
    )
```

A different initialization scheme can be defined by overwriting this method with the desired scheme. See [the FeedforwardController of the copy task](../tasks/copy/feedforward_controller.py) as an example of different initialization scheme.

### Using the DNC module

Once you defined your concrete controller class, you're then ready to plug in that controller and use the DNC on your task.

To do that, you need to construct an instance of the DNC module and pass it your controller class and the desired parameters of your model. The constructor of the DNC module is defined as follows:

```python
DNC.__init__(
    controller_class,
    input_size,
    output_size,
    max_sequence_length,
    memory_words_num = 256,
    memory_word_size = 64,
    memory_read_heads = 4,
    batch_size = 1
)
```
* **controller_class**: is a reference to the concrete controller class you defined earlier. You just need to pass the class, you do not need to construct an instance yourself; the `DNC` constructor will automatically handle that.
* **input_size**: the size of the flatten input vector.
* **output_size**: the size of the flatten output vector.
* **max_sequence_length**: the maximum length of input sequences that is expected to be fed into the model.
* **memory_words_num**: the number of memory locations.
* **memory_word_size**: the size of an individual memory location.
* **memory_read_heads**: the number of read head in the memory.
* **batch_size**: the size of the batch to be fed to the model.

As you may have noticed, you do not construct an instance of `Memory` directly, you just pass the desired parameters and the `DNC` module will handle its construction.

To get define the operations leading to the output out of the model, you use the instance method `get_outputs()`:
```python
output_op, memory_view = dnc_instance.get_outputs()
```
*`memory_view` is a pyton `dict` that carries some of the internal values of the model (like weightings and gates) that is mainly used for visualization.*

To actually get the outputs, you need to run this `output_op`, while feeding three placeholders that are attributes of the dnc instance. These placeholders are:
* **input_data**: a 3D tensor of shape `batch_size X sequence_length X input_size` which represents the inputs of that run.
* **target_output**: a 3D tensor of shape `batch_size X sequence_length X output_size` which represents the desired outputs.
* **sequence_length**: a integer that define the sequence length across that batch. **Notice** that this means that the whole batch must be of the same sequence length (which is a to-be-addressed limitation), but sequence_length can vary between batches as long as they are less than or equal to the `max_sequence_length` the DNC was instantiated with.

So a run for an instantiated DNC model looks like:
```python
input_data = ...
target_output = ...
sequence_length = 10

output = dnc_instance.get_outputs()
loss = some_loss_fn(output, dnc_instance.target_output)

loss_val, dnc_output = session.run([loss, output], feed_dict={
    dnc_instance.input_data: input_data,
    dnc_instance.target_output: target_output,
    dnc_instance.sequence_length: sequence_length
})
```
After you train your model, you can save a check point to disk using the `save` method. This method takes three arguments: the tensorflow session, the path to the checkpoints directory at which the checkpoint will be saved, and the name to be saved with.
```python
dnc_instance.save(session, './checkpoints_dir', 'checkpint-1')
```
To restore a previous check point, you can use the `restore` method with the 1st two parameters like in saving and the 3rd one is now the name of the existing checkpoint to be restored.
```python
dnc_instance.restore(session, './checkpoints_dir', 'checkpint-1')
```
#### An Example

The following is an excerpt from the copy task trainer to demonstrate how a `DNC` instance can be integrated with an optimizer to construct a complete graph.
```python
optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)

ncomputer = DNC(
    FeedforwardController,
    input_size,
    output_size,
    2 * sequence_max_length + 1,
    words_count,
    word_size,
    read_heads,
    batch_size
)

# squash the DNC output between 0 and 1
output, _ = ncomputer.get_outputs()
squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)

loss = binary_cross_entropy(squashed_output, ncomputer.target_output)

gradients = optimizer.compute_gradients(loss)
for i, (grad, var) in enumerate(gradients):
    if grad is not None:
        gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

apply_gradients = optimizer.apply_gradients(gradients)
```
