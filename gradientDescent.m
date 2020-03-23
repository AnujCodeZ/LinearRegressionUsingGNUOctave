% Function to calculate gradient descent.
% In this we simply looping over num_times 
% and reducing theta simultaneously.
% To record cost use j_hist and upgrade it over every iteration.

function [j_hist,theta] = gradientDescent(X,y,theta,alpha,num_times)

	j_hist = zeros(num_times,1); % Initializing j_hist.
	m = size(X, 1);
	n = size(theta,1);

	for i=1 : num_times
		pred = X*theta;
		for j = 1 : n
	    	theta(j) = theta(j) - alpha*(1/m)*sum((pred - y).*X(:,j));
	    end
		j_hist(i) = costFunction(X,y,theta);
	end

end