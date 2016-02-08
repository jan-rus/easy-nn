###
Returns standard dot product (scalar product) of two
vectors of scalars. Vectors have to have equal length.


@param {Array<Number>} a vector of scalars
@param {Array<Number>} b vector of scalars
@return {Number} dot product of the given vectors
###
dot = (a, b)->
	res = 0
	res += a[i]*b[i] for i in [0...a.length]
	res

###
Special case of the logistic function, params from (-inf; +inf),
results from (0;1). Used as an activation function.

@param {Number} value input to the sigmoid function
@return {Number} result of the sigmoid function
###
sigmoid = (x)->
	1 / (1 + Math.exp(-1*x))


###
Calculates mean square error for two vectors of scalars.
One of them contains real values, another one contains
desired values.

@param {Array<Number>} a vector of scalars
@param {Array<Number>} b vector of scalars
@return {Number} mean square error
###
vectorMSE = (a, b)->
	totalError = 0
	totalError += (b[i]-a[i]) * (b[i]-a[i]) for i in [0...b.length]
	totalError / b.length

###
Copies values from array "a" to array "b". Arrays
should have the same length or at least the "a" array
should be longer than the "b" array.

@param {Array<Number>} a array we read values from
@param {Array<Number>} B array we write values into
###
copyArray = (a, b)->
	b[i] = a[i] for i in [0...a.length]

###
Clamps given value when close to 1.0 or 0.0,
returns -1 when not close to them.

@param {Number} value to clamp
@return clamped value 
###
clamp = (x)->
	return 1 if x > 0.9
	return 0 if x < 0.1
	x

###
Generates random integer smaller than given number.

@param {Number} limit limit for generated number
@return random integer
###
randInt = (limit)->
	Math.floor(limit * Math.random())

###
Shuffles given data

@param {Array<Number>} arr array of values that should be shuffled
###
shuffle = (arr) ->
	return arr[randInt(arr.length)] if arr.length <= 1

	for i in [0...arr.length]
		index = randInt(i+1)
		[arr[index], arr[i]] = [arr[i], arr[index]]

	arr

module.exports.dot = dot
module.exports.sigmoid = sigmoid
module.exports.vectorMSE = vectorMSE
module.exports.copyArray = copyArray
module.exports.clamp = clamp
module.exports.shuffle = shuffle