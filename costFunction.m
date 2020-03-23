% Function to calculate cost function.
% It mainly helps in gradient descent.
% In this we calculate squared mean error.

function J = costFunction(X,y,theta)

	prediction = X*theta;
	m = size(X, 1);
	sqError = (prediction - y).^2; % Square to ignore negative values.
	J = (1/(2*m))*sum(sqError); % By 2m because in derivative of square it cancels out.

end