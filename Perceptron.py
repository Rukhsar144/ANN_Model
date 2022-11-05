class Perceptron:

	def __init__(self, weights=[1,1], bias=-2, lr=0.1):
		self.__weights = weights
		self.__bias = bias
		self.__learning_rate = lr
		self.__accuracy = 0
		
	def predict(self, X):
		y = self.__feed_forward(X)
		return y
		
	def __feed_forward(self, X):
		x = self.__sum(X)
		return self.__step_function(x)
		
	def __sum(self, X):
		total = self.__bias
		for i, x in enumerate(X):
			w = self.__weights[i]
			total += x*w
		return total
		
	def __step_function(self, x):
		if x >= 0:
			return 1
		else:
			return 0
		
	def train(self, train_data, labels):
		for i, X in enumerate(train_data):
			y = labels[i]
			y_predict = self.__feed_forward(X)
			self.__update(X, y_predict, y)
			self.print_accuracy(y_predict, y, i+1)		
			
	def __update(self, X, y_predict, y):
		factor = self.__learning_rate * (y_predict - y)
		self.__bias -= factor
		for i in range(len(self.__weights)):
			self.__weights[i] -= factor * X[i]	

	def print_accuracy(self, y_predict, y, processed_examples):
		if y_predict - y == 0:
			self.__accuracy += 1
		
		ma = (self.__accuracy / processed_examples)*100		
		print("Accuracy of the model is: " +str(round(ma, 2))+ "%")
			
	def print_weights(self):
		print("Bias: ", self.__bias)
		print("Weights: ", self.__weights)
