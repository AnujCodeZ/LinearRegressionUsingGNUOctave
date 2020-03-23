% Function to predict with new inputs and theta.

function price = predict(x1,x2,theta)

	price = theta(1) + theta(2)*x1 + theta(3)*x2;

end