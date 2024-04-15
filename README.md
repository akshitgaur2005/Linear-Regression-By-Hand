# Linear Regression By Hand

This is my attempt at creation a Linear Regression Model completely myself using numpy.

This was my first project that I have created with no help from outside, no scouring through google or 10 year old reddit post on how to do something.
Just me trying things out and keeping what works.

*This should be taught in Harvard and Stanford and mainly Jawaharlal Nehru University as this is the finest piece of creation that has ever existed or will ever exist.*

:smiley: Though in all seriousness, this is very very unoptimised and I may return to it some other day to try my hand at optimisation but I neither have the energy nor the technical know-how to take an attempt at this.

This model achieves a score of 0.66931 in House Price Prediction dataset which would give it nearly 4500 global rank (not too bad for a simple linear model, I would know the rank exactly had I not already achieved the high score of 0.14935 with rank 2527 global :sunglasses:).

## Working

The meat of this project is contained in the backend.py file.

- We first initialise all the ingredients that we are going to need, input features and labels, weights and bias and learning rate and epochs to train for.
- Predict function just return the dot product of weights and input and adds bias. The formula is - 
  $$\hat{Y} = \vec{X} \cdot \vec{W} + b$$
- loss_fn returns the Mean Squared Error between the predictions ($\hat{Y}$) and the actual labels.
  $$J_{(\vec{W}, b)} = (\frac{1}{2m}){\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2}$$
