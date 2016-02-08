math = require "./math.coffee"

class NeuralNetworkTrainer

	# default options
	learningRate: 0.1
	momentum: 0.8
	maxEpochs: 20000
	minAccuracy: 0.95
	debug: no

	# handling of validation
	epochs: 0
	trainingAccuracy: 0 # accuracy for training data, stopping condition
	generalizationAccuracy: 0 # accuracy for unseen data, stopping condition
	validationAccuracy: 0 # accuracy for validation data, output of the training

	###
	Creates new trainer for neural networks

	@param {Object} neuralNetwork network that should be trained
	@param {Object} opts object with training options
		learningRate: [0.0;1.0], rate of learning, not required, default is set to 0.5, recommended [0.05;2.0]
		momentum: [0.0;1.0], momentum of weight updates, recommended [0.0;1.0]
		maxEpochs: [0;+inf], max epochs of training to stop training
		minAccuracy: [0.0;1.0], min accuracy of the network to stop training
		debug: [true/false], set to true to see details about training online
	###
	constructor: (neuralNetwork, opts)->
		@nn = neuralNetwork
		@setOptions opts

	###
	Sets options of the training.

	@param {Object} opts object with training options
		learningRate: [0.0;1.0], rate of learning, not required, default is set to 0.5, recommended [0.05;2.0]
		momentum: [0.0;1.0], momentum of weight updates, recommended [0.0;1.0]
		maxEpochs: [0;+inf], max epochs of training to stop training
		minAccuracy: [0.0;1.0], min accuracy of the network to stop training
		debug: [true/false], set to true to see details about training online
	###
	setOptions: (opts)->
		@[key] = val for key, val of opts

	###
	Shufles given dataset and slits the dataset into training,
	generalization and validation set of data.

	@param {Array<Number>} inputs set of inputs for neural network
	@param {Array<Number>} outputs set of outputs from neural network
	@return {Array<Array>} array with individual test sets
	###
	inds: null
	_splitTrainingData: (inputs, outputs)=>
		trainingSetIn = [] # should contain 60% of data
		trainingSetOut = [] # should contain 60% of data
		generalizationSetIn = [] # should contain 20% of data
		generalizationSetOut = [] # should contain 20% of data
		validationSetIn = [] # should contain 20% of data
		validationSetOut = [] # should contain 20% of data

		indices = []
		unless @inds
			indices =  [0...inputs.length].map (i)-> i
			math.shuffle indices
			@inds = indices
		else
			indices = @inds

		for i in [0...indices.length]
			tmp = i%5
			switch
				when tmp is 0
					validationSetIn.push inputs[indices[i]]
					validationSetOut.push outputs[indices[i]]
				when tmp is 1
					generalizationSetIn.push inputs[indices[i]]
					generalizationSetOut.push outputs[indices[i]]
				when tmp >= 2
					trainingSetIn.push inputs[indices[i]]
					trainingSetOut.push outputs[indices[i]]

		[trainingSetIn, trainingSetOut, generalizationSetIn, generalizationSetOut, validationSetIn, validationSetOut]

	###
	Validates the neural network such that compares outputs with
	desired outputs and counts them. Finally calculates accuracy
	on the validation set.

	@param {Array<Number>} inputs array of inputs
	@param {Array<Number>} desiredOutputs array of desired outputs
	@return accuracy
	###
	_validateOutputs: (inputs, desiredOutputs)->
		correct = 0
		for i in [0...inputs.length]
			output = @nn.feedForward inputs[i]

			if output[0] > output[1]
					output[0] = 1
					output[1] = 0
			else if output[0] < output[1]
					output[0] = 0
					output[1] = 1
			else
				console.log output

			isSame = yes
			for j in [0...desiredOutputs[i].length]

				isSame = no if math.clamp(output[j]) isnt desiredOutputs[i][j]
			correct++ if isSame

		correct / inputs.length

	###
	Runs training of the given neural network, stops when stopping
	conditions are met. Requires at lest 5 paterns to be fed in.

	@param {Array<Number>} inputs array of inputs
	@param {Array<Number>} desiredOutputs array of desired outputs
	@return accuracy
	###
	train: (inputs, outputs)=>
		[tSetIn, tSetOut, gSetIn, gSetOut, vSetIn, vSetOut] = @_splitTrainingData inputs, outputs
		
		@epochs = 0
		if @debug then console.log "Epochs;Training Accuracy;Generalization Accuracy"
		while true
			@epochs++
			for i in [0...tSetIn.length]
				output = @nn.feedForward tSetIn[i]

				if math.vectorMSE(output, tSetOut[i]) > 0
					@_prepareNewWeightsOutput @nn.layers[@nn.layers.length-1], tSetOut[i]
					
					for l in [@nn.layers.length-2..0]
						@_prepareNewWeightsHidden @nn.layers[l], @nn.layers[l+1]

					for l in [0...@nn.layers.length]
						@_setNewWeights @nn.layers[l]

			@trainingAccuracy = @_validateOutputs tSetIn, tSetOut
			@generalizationAccuracy = @_validateOutputs gSetIn, gSetOut
			
			if @debug then console.log "#{@epochs};#{@trainingAccuracy};#{@generalizationAccuracy}"
			break if (@generalizationAccuracy > @minAccuracy and @trainingAccuracy > @minAccuracy) or (@epochs >= @maxEpochs)

		# final validation on validation set
		@validationAccuracy = @_validateOutputs vSetIn, vSetOut

	###
	Backpropagation, calculates errors and new weights for output layer.

	@param {Object} outputLayer output layer
	@param {Array<Number>} desiredOutput array with desired output of the output layer
	###
	_prepareNewWeightsOutput: (outputLayer, desiredOutput)=>
		for i in [0...outputLayer.weights.length] # for each neuron
			outputLayer.errorGradients[i] = (desiredOutput[i] - outputLayer.output[i]) * outputLayer.output[i] * (1 - outputLayer.output[i]) # output error gradient
			for j in [0...outputLayer.weights[i].length] # for each weight
				outputLayer.weightCorrectionDelta[i][j] = @learningRate * outputLayer.errorGradients[i] * outputLayer.input[j] + @momentum * outputLayer.weightCorrectionDelta[i][j]
				outputLayer.weightsNew[i][j] = outputLayer.weights[i][j] + outputLayer.weightCorrectionDelta[i][j]

	###
	Backpropagation, calculates errors and new weights for given hidden layer.

	@param {Object} layer hidden layer that will be processed
	@param {Object} layerDeeper hidden layer necessary for processing (the one closer to the output)
	###
	_prepareNewWeightsHidden: (layer, layerDeeper)=>
		for i in [0...layer.weights.length] # for each neuron
			weightedSum = 0
			for j in [0...layerDeeper.weights.length] # for each neuron in layer closer to output
				weightedSum += layerDeeper.weights[j][i] * layerDeeper.errorGradients[j]
			layer.errorGradients[i] = weightedSum * layer.output[i] * (1 - layer.output[i])

			for j in [0...layer.weights[i].length] # for each weight
				layer.weightCorrectionDelta[i][j] = @learningRate * layer.errorGradients[i] * layer.input[j] + @momentum * layer.weightCorrectionDelta[i][j]
				layer.weightsNew[i][j] = layer.weights[i][j] + layer.weightCorrectionDelta[i][j]

	###
	Backpropagation, overwrites weights of given layer with new precalculated weights.
	
	@param {Object} layer layer that will be processed
	###
	_setNewWeights: (layer)->
		for i in [0...layer.weights.length] # for each neuron
			layer.weights[i] = layer.weightsNew[i]

module.exports.NeuralNetworkTrainer = NeuralNetworkTrainer