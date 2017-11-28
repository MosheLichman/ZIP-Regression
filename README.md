# ZIP-Regression
My code for fixed-effect Zero-Inflated Poisson Regression model.

This code is part of my work on the Prediction of Sparse User-Item Consumption Rate.

As such, it means that it's an academic code. I've done a lot of work to make it super efficient and I believe there are more 
comments in the code than actual code, but I'm sure more could have been done.

## Example Notebooks
To make it easier to work with the code, I added to IPython notebooks to the repository. 

1. The first one creates a synthetic data that I've used for uni-testing, but you can use it to understand the formatting of the data.
2. The second one is a complete run example of the model using the synthetic data showing both sanity uni-tests and empirical evaluations.

## Dependencies
1. numpy and scipy
2. pandas
3. cython
4. You also need to run the setup.py. Type in the cmd line 'python setup.py build_ext --inplace' in the project directory.
