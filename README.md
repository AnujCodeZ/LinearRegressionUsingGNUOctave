I implement the linear regression from scratch in Octave.
It includes all the mathematics to understand the working of gradient descent.
By implementing this I clearly understand all the concepts of linear regression,
which is the basis of machine learning.

MOTIVATION.

In today's machine learning world there are a lot of libraries. Using them is very easy and very helpful to the industry level.
But, by using them you'll never understand the concepts behind the learning algorithms. That how they work actually. Because, if you don't go deeper then there is no catch in that, you eventually get bored and start unliking it as I do.
So, building from scratch is always fun and interesting to know the things behind it. Unless you don't like mathematics, then you are in the wrong field.

GNU OCTAVE.

Octave is an open-source programming language.
It turns out building machine learning algorithms from scratch is easy in this language. Because it is more math friendly and fast.
If you want to scale up. First, build in octave then go to other programming languages like Python.

LINEAR REGRESSION.

Just to make it simple, I categories the building of the machine learning model into 5 steps. This is a simple model of house price prediction.

STEP 1: Getting data.

First, we collect data on a real-world example. In this problem, we have 27 examples of houses. The area in sqft., the number of rooms as features and prices as the labels.
Features: The independent variables. (Input)
Labels: The dependent variables. (Output)

STEP 2: Preprocessing.

Generally, there are various processes. But, in this particular problem, we only need feature scaling. In this process, we scale the value of the features in a finite range (-1 and 1).

FORMULA:

	mu = mean(X);
	s = max(X) - min(X);
	X := (X - mu)./s;

STEP 3: Cost Function.

The cost function is the measure of how much our model is wrong. We use MSE (Mean squared error). Use square to ignoring negative values. Divide by 2m instead of m (m is the number of training examples). Because when calculating the derivative of error 2 cancels out.

FORMULA:

	prediction = X*theta;
	error = (prediction - y).^2;
	J = (1/(2*m))*sum(error);

Theta is an array of numbers of the size one greater than the number of features.

STEP 4: Gradient Descent.

Gradient descent is a legend algorithm for optimizing the cost function. It decreases the value of theta in accordance with minimizing the cost function.

FORMULA:

	for i=0:epochs
		prediction = X*theta;
		theta = theta - alpha*(1/m)*sum((prediction - y).*X);
		j_history(i) = J(X,y,theta);
	end

epochs are the num of times we want to repeat the training.
j_history is the record of cost function over the iterations.
Plotting j_history with the number of iterations gives a decreasing curve.

STEP 5: Prediction.

Our theta is now modified by using it we can predict now. But first, we also scale the new features and then add 1 at the start of the new features array.

FORMULA:

	prediction = X*theta;
	
X here is array of new features including 1 at start.

CONCLUSION.

Now you know what to do in general ML problems. Let's get going with scratch. This post not explaining the exact mathematics. You may be confused. It is the only generalization of the process of making a machine learning model.
To better understanding go through the course of machine learning on Coursera.

RESOURCE.

Machine Learning course: https://www.coursera.org/learn/machine-learning

	THANKS, AND KEEP CODING
