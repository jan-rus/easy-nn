math = require "./math.coffee"

class NeuralNetwork

	###
	Creates new "feedforward neural network" with given topology and learning rate

	@param {Array<Number>} topology ...
	###
	constructor: (topology)->
		@values = []
		@layers = []

		# prepare buffers for inputs/outputs of layers output of one hidden
		# layer is input of next layer including output layer every layer has
		# one more input - this is for the bias neuron
		@values.push([0...neuronCount+1].map ()-> 1) for neuronCount in topology

		@input = @values[0]
		@output = @values[@values.length-1]

		# create layers for hidden layers and output layer
		@layers.push(@_createLayer(@values[i], @values[i+1])) for i in [0...topology.length-1]

	###
	Feed pattern through network and return results.

	@param {Array<Number>} input one piece of input/pattern
	@result {Array<Number>} result of the neural network
	###
	feedForward: (input)=>
		math.copyArray input, @input # load input into input layer
		@_processInput(layer) for layer in @layers
		@output[0...@output.length-1] # return the result of the feed forward that is on the output of the network

	###
	Creates structure for one layer with given input and output size

	@param {Array<Number>} inputBuffer its size determines dimension of the input of the layer
	@param {Array<Number>} outputBuffer its size number of neurons in the layer
	@return {Object} laer of the neuron network
	###
	_createLayer: (inputBuffer, outputBuffer) ->
		inSize = inputBuffer.length
		outSize = outputBuffer.length
		layer =
			input: inputBuffer
			weights: [0...outSize-1].map ()-> [0...inSize].map ()-> (Math.random() - 0.5)
			output: outputBuffer
			errorGradients: [0...outSize-1].map ()-> 0
			weightCorrectionDelta: [0...outSize-1].map ()-> [0...inSize].map ()-> 0
			weightsNew: [0...outSize-1].map ()-> [0...inSize].map ()-> 0.5

	###
	Processes input of the layer and sets result to its output.

	@param {Object} layer layer that should be processed
	###
	_processInput: (layer)->
		for i in [0...layer.output.length-1]
			layer.output[i] = math.sigmoid(math.dot(layer.input, layer.weights[i]))

module.exports.NeuralNetwork = NeuralNetwork