# easy-nn

The __easy-nn__ is "easy to use" lightweight implementation of a __feed forward neural network__.
The network has to consist of at least one neuron, so called _perceptron_. Each neuron of the network
counts with a bias, which is automatically generated and is not a part of the topology deffinition.
It is possible to store and load a state of the network and of course, you can train it.

## Installation

	npm install easy-nn

## Examples

### Creating Neural Network

A simple example that shows how to create a neural network with given topology. The topology
is defined as a set of layers, each layer has predefined number of neurons. The input
and output layers always have to be defined, with at least one neuron. Hidden layers are optional.
Every neuron weights all values from previous layer, insluding the bias.

```coffeescript
{NeuralNetwork} = require "easy-nn"

# Each input has 1 value, i.e. input layer has 1 entry,
# output layer has 1 neuron
perceptron1 = new NeuralNetwork [1, 1]

# Each input has 3 values, i.e. input layer has 3 entries,
# output layer has 1 neuron
perceptron2 = new NeuralNetwork [3, 1]

# Each input has 74 values, i.e. input layer has 74 entries,
# 1st hidden layer has 14 neurons,
# output layer has 2 neurons
network1 = new NeuralNetwork [74, 14, 2]

# Each input has 21 values, i.e. input layer has 21 entries,
# 1st hidden layer has 7 neurons,
# 2nd hidden layer has 9 neurons,
# output layer has 3 neurons
network2 = new NeuralNetwork [21, 7, 9, 3]
```

Previous code results in four neural networks with randomly set weights for each neuron.
Each set of weights for neuron inputs contains weight for bias as well. The bias is
implemented as _invisible_ neuron in previous layer outputting constant 1. The first two
neural networks are so called _perceptrons_, the simplest neural networks with only one neuron.


### Using Neural Network

Once you have created a network, you can easily feed some input in and get a response
of the neural network. Every input is an array of values and every output is an array
of values, one value for each neuron.

```coffeescript
{NeuralNetwork} = require "easy-nn"

# Perceptron that processes only one value on input
perceptron = new NeuralNetwork [1, 1]
output = perceptron.feedForward [6]
console.log output # prints array with 1 value

# Network that processes 4 values on input
network = new NeuralNetwork [4, 3, 2]
output = network.feedForward [11, 7, 9, 5]
console.log output # prints array with 2 values
```

Of course, responses (outputs) in this example contain nonsenses, because the networks
are not trained yet.

### Training Neural Network

You can use a neural network trainer to train the network. One trainer is built for
one specific network, but it is configurable and can be used to train the network repeatedly
with different data or settings. The trainer modifies weights of the original network and
once the training is over, returns accuracy of the trained network.

```coffeescript
{NeuralNetwork, NeuralNetworkTrainer} = require "easy-nn"

nn = new NeuralNetwork [2, 1]
nnTrainer = new NeuralNetworkTrainer nn

# inputs array is set of points, each with {x, y} coordinates
# desired response is 0 if x-coordinate is < 4, else 1
inputs = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 8], [6, 2]]
outputs = [[0], [0], [0], [1], [1], [1]]

# train the network using inputs and desired responses/outputs
acc = nnTrainer.train inputs, outputs

console.log "Network trained with #{(acc * 100).toFixed(2)}% accuracy after #{nnTrainer.epochs} epochs.\n"

# the network is trained now, lets use it as it is and check the results
for i in [0...inputs.length]
	output = nn.feedForward inputs[i]
	console.log "input: #{inputs[i]}, output: #{output}, desired output: #{outputs[i]}"
```
Output should look similar to this:

	Network trained with 100.00% accuracy after 1045 epochs.

	input: 1,2, output: 0.0031683220517143684, desired output: 0
	input: 2,3, output: 0.08543480587781929, desired output: 0
	input: 3,1, output: 0.09277863465043776, desired output: 0
	input: 4,3, output: 0.9000008820059654, desired output: 1
	input: 5,8, output: 0.9999529756701503, desired output: 1
	input: 6,2, output: 0.9965586497259523, desired output: 1

Of course, the training took too many epochs, because the trainer is set to default. There
are few options you can set, that should help with that:

* Training Parameters:
	* __learningRate__: learning rate (related to size of change of weights), default is 0.1
	* __momentum__: momentum (related to dumping of change of weights), default is 0.8
	* __maxEpochs__: max number of epochs until the training is stopped, default is 20,000
	* __minAccuracy__: min accuracy that has to be met to stop the training, default is 0.95
	* __log__: writes number of epochs and accuracy into console, default is false
	* __logPeriod__: writes only every logPeriod-th log line into console, default is 1

```coffeescript
{NeuralNetwork, NeuralNetworkTrainer} = require "easy-nn"

nn = new NeuralNetwork [2, 1]
nnTrainer = new NeuralNetworkTrainer nn, {learningRate: 0.6, momentum: 0.8}

# or we can set some options this way
nnTrainer.setOptions {maxEpochs: 500, minAccuracy: 0.9}

# inputs array is set of points, each with {x, y} coordinates
# desired response is 0 if x-coordinate is < 4, else 1
inputs = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 8], [6, 2]]
outputs = [[0], [0], [0], [1], [1], [1]]

# train the network using inputs and desired responses/outputs
acc = nnTrainer.train inputs, outputs

console.log "Network trained with #{(acc * 100).toFixed(2)}% accuracy after #{nnTrainer.epochs} epochs.\n"

# the network is trained now, lets use it as it is and check the results
for i in [0...inputs.length]
	output = nn.feedForward inputs[i]
	console.log "input: #{inputs[i]}, output: #{output}, desired output: #{outputs[i]}"
```

Output should look similar to this:

	Network trained with 100.00% accuracy after 95 epochs.

	input: 1,2, output: 0.003421933296556123, desired output: 0
	input: 2,3, output: 0.08859437917377035, desired output: 0
	input: 3,1, output: 0.09758939010565129, desired output: 0
	input: 4,3, output: 0.9000483968432926, desired output: 1
	input: 5,8, output: 0.9999475925358006, desired output: 1
	input: 6,2, output: 0.9964863616017431, desired output: 1

### Storing & Loading Neural Network

Once the neural network is created and trained it is quite handy to be able to store the
structure and state of the network for later use. It is provided in a form of JSON string.

```coffeescript
{NeuralNetwork, NeuralNetworkTrainer} = require "easy-nn"

nn = new NeuralNetwork [2, 1]
nnTrainer = new NeuralNetworkTrainer nn

# inputs array is set of points, each with {x, y} coordinates
# desired response is 0 if x-coordinate is < 4, else 1
inputs = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 8], [6, 2]]
outputs = [[0], [0], [0], [1], [1], [1]]

# train the network using inputs and desired responses/outputs
acc = nnTrainer.train inputs, outputs

console.log "Network trained with #{(acc * 100).toFixed(2)}% accuracy after #{nnTrainer.epochs} epochs.\n"

nnState = nn.toJSON()

console.log nnState

```

The _nnState_ string contains information about topology of the network, where _layers_ object
contains a key for each layer and each layer is represented by array of neurons. Every neuron
has an array of weights for its inputs, including weight for bias - that is the last one.
Input layer doesn't contain weights, there are no neurons, just inputs. Output should look like this:

	Network trained with 100.00% accuracy after 1045 epochs.
	{
		"layers": {
			"input": [
				[],
				[]
			],
			"output": [
				[
					2.284898341232388,
					1.0984011751180214,
					-10.237093429332361
				]
			]
		}
	}


The neural network can be set to a state stored in a JSON string during initialisation
or an existing neural network can be reconfigured into the new topology and state
using the JSON string. Original topology and state will be lost.

```coffeescript
{NeuralNetwork, NeuralNetworkTrainer} = require "easy-nn"

inputs = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 8], [6, 2]]
outputs = [[0], [0], [0], [1], [1], [1]]


nn = new NeuralNetwork [2, 1]

state1 = nn.toJSON() # not trained network stored as string

nnTrainer = new NeuralNetworkTrainer nn
nnTrainer.train inputs, outputs

state2 = nn.toJSON() # trained network stored as string

# if you want to, you can store the sate into file now
# and load it later and restore the network

restoredNN = new NeuralNetwork state2 # network initialised from string

# or reset existing network

restoredNN.fromJSON state1 # network reconfigured from string
```