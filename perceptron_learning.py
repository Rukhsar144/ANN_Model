
from perceptron import Perceptron

X = [[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,0],[0,1],[1,0]]

y= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

nn = Perceptron()
print("Below are the deafult weights before training:")
nn.print_weights()

print(">>starting training...")
nn.train(X, y)
print(">>Training finished")

pred_example = [1,1]

y_predict = nn.predict(pred_example)

print("\nAND-logic prediction = ", y_predict)
print("Below are the adjusted weights after training:")
nn.print_weights()
