math = require "./math.coffee"

class NeuralNetwork

	###
	Creates new "feed forward neural network" with given topology and learning rate

	@param {Array<Number> or String} topologyOrState use array to describe only
		topology of the network or JSON string representation of the network
	###
	constructor: (topologyOrState)->
		if typeof topologyOrState is "string"
			@fromJSON topologyOrState
		else
			@_createNetworkStructure topologyOrState

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
	Builts structure of the neural network, buffers for inputs/outputs of layers,
	output of one hidden layer is input of next layer. Including output layer,
	every layer has one more input - this is for the bias neuron.
	
	@param {Array<Number>} topology description of topology of the neural network
	###
	_createNetworkStructure: (topology)->
		@values = []
		@layers = []

		@values.push([0...neuronCount+1].map ()-> 1) for neuronCount in topology

		@input = @values[0]
		@output = @values[@values.length-1]

		# create layers for hidden layers and output layer
		@layers.push(@_createLayer(@values[i], @values[i+1])) for i in [0...topology.length-1]

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

	###
	Converts structure and state of the neural network into JSON string.

	@return {String} string representation of the neural network
	###
	toJSON: () =>
		network = {layers: {}}
		network.layers.input = [0...@layers[0].weights[0].length-1].map ()-> []
		network.layers["hidden#{i+1}"] = @layers[i].weights for i in [0...@layers.length-1]
		network.layers.output = @layers[@layers.length-1].weights

		JSON.stringify network, null, "\t"

	###
	Builds structure and state of the neural network according to given
	JSON string.

	@param {String} string representation of the neural network
	###
	fromJSON: (strConfig) =>
		nn = JSON.parse strConfig
		hLayerCount = Object.keys(nn.layers).length - 2

		topology = [] # only topology description
		topology.push nn.layers.input.length
		topology.push nn.layers["hidden#{i+1}"].length for i in [0...hLayerCount]
		topology.push nn.layers.output.length
		
		@_createNetworkStructure topology # initialises structure of the network

		for i in [0...hLayerCount] # sets state of the hidden layers
			for j in [0...@layers[i].weights.length]
				@layers[i].weights[j] = nn.layers["hidden#{i+1}"][j].slice()

		for j in [0...@layers[@layers.length-1].weights.length]  # sets state of the output layer
			@layers[@layers.length-1].weights[j] = nn.layers.output[j].slice()

module.exports.NeuralNetwork = NeuralNetwork