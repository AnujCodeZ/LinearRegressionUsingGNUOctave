% There are 5 basic steps to do any type of machine learning problem
% I implement the linear regression from scratch including the maths using these steps

% STEP 1 : Getting data
% Here we load data into X and y
% X : features (independent)
% y : labels (dependent on features) 

X = load('featuresX.dat');
y = load('priceY.dat');

% STEP 2 : Feature engineering
% In this step we analyze data and do some manipulation in data
% to get better results.
% One method is feature scaling, in this we bring all data in the range of -1 and 1.

[X,mu,s] = featureScaling(X);

% STEP 3 : Cost Function
% Cost function is show us that how wrong is our model.
% And we try to reduce this function.

m = size(X, 1); % Number of rows.
n = size(X, 2); % Number of columns.

X = [ones(m, 1) X]; % Add columns of ones to add biasing.

theta = zeros(n+1,1); % Initializing theta.

J = costFunction(X,y,theta);

% STEP 4 : Gradient Descent
% Gradient descent is a optimization algorithm.
% It helps in reducing value of theta according to the cost.

alpha = 2.01; % alpha is step by which we reduce theta.
num_times = 100; % Number of times we repeat training, or we can say epochs.

% We get two parameters j_hist and theta.
% j_hist is recording the cost in every epochs.
[j_hist, theta] = gradientDescent(X,y,theta,alpha,num_times);

% Plotting j_hist with iterations. If it is decreasing then our model is right.
plot(1:numel(j_hist), j_hist, '-b', 'LineWidth', 2);
xlabel("Number of iterations");
ylabel("Cost function J");

% STEP 5 : Predicting
% Get data and predict through model.

x1 = input("Enter area of house : ");
x2 = input("Enter no. of bedrooms : ");

% We use mu and s to scale the new input.
x1 = (x1 - mu(1))./s(1);
x2 = (x2- mu(2))./s(2);
price = predict(x1,x2,theta);
disp("price of the house is : ");
disp(price);

