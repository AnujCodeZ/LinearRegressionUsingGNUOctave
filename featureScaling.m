% Function to scale features.
% In this we subtract mean and divide by range.
% This calculation makes the features in the range of -1 and 1.

function [x,mu,s] = featureScaling(x)

	mu = mean(x); % Mean.
	s = max(x) - min(x); % Range.

	x = x - mu;
	x = x ./ s;

end