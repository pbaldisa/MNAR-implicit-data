# Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback
Understanding the article of the same name by Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata and Kazuhide Nakata. The full article can be foudn [here](https://arxiv.org/pdf/1909.03601.pdf).

The article by Yuta Saito et. al. analyses two existing methods for implicit feedback recommendation, showing their shortcomings. Then, it introduces a new method based on the use of a different loss function and a new estimator. One of the most relevant accomplishments of this study is to change the paradigm of what one optimises. Instead of predicting items with the highest click probability, one should predict the items with the highest relevance.

In this post, I intend to understand why the existing baselines can be improved using the proposed method. With this objective in mind, we will dive into some key concepts like: what is a latent probabilistic model, why Saito’s new loss function is better suited for the action at hand, and why the new method is closer to solving the two main problems of implicit feedback recommendation: MNAR and postive-unlabeled problem.

# Introduction

The current importance of recommender systems is undoubted. They are used in a spectrum of cases from personalized advertising to the next video that shows on your “For You Page”. These systems can use two types of data: explicit and implicit data. 

**Explicit data** includes users’ ratings on items. Think of the stars you award to a certain book in GoodReads or the rating of a purchase in Amazon. This represents the preferences explicitly, but it is more invasive and harder to acquire. 

On the other hand, **implicit data** corresponds to users’ clicks (views, purchases).  More formally, observations of a user’s actions that are interpreted as statements on the relevance of a particular item [[Book Chapter]](https://link.springer.com/chapter/10.1007/978-3-319-90092-6_14). From now on, the data we will work with is click data i.e. wether a user has interacted with an item.

# Positive-unlabeled problem

Working with implicit data can lead to what is called the **********positive-unlabeled problem**********. Since implicit data is not directly tied to the user’s preference, one cannot know whether unclicked feedback is negative or unlabeled positive feedback. For example, the user might have not been exposed to an item which they would like. This item would be unclicked, but relevant. 

Previous solutions to this problem include:

- Uniformly upweighting loss for positive feedback data. **Weighted Matrix Factorization (WMF)** assigns less weight to unclicked items to incorporate the idea that they correspond to lesss confidence in prediction. However, this is not always the case: unclicked items to which the user has been repeatedly exposed to are less relevant.
- Estimating confidence of data having relevance information. **Exposure Matrix Factorization (ExpoMF)** uses exposure information to estimate the confidence of the predictions. It uses a latent probabilistic method in which the probability of click is proportional to the probabilities of exposure and relevance. If a user has been exposed to an item, clicked or uncliked behaviour can be seen as relevance. Therefore, the exposure probability is regarded as the confidence of how much relevance information each data includes.

ExpoMF tackles the positive-unlabeled problem. Despite this, it does not address the missing-not-at-random (MNAR) problem: by upweighting the loss of data with high exposure probability, it can leed to poor prediction for rare items.

# Missing-not-at-random problem

This problem arises because users are more likely to click on popular items, even if those are not as relevant to their preferences. Think of all the times you have watched a TV show because everyone was talking about it, even if it wasn’t a genre you normally go for. 

The new method aims to solve this problem by changing what the recommender systems predicts. Instead of predicting click probability, the method predicts the item with the highest relevance. This allows it to simultaneously solve the positivie-unlabeled problem and the MNAR.

To do so, they were inspired by the estimation technique for casual inference, and derived an unbiased estimator for the ideal loss corresponding to item relevance. Let’s start by understanding what casual inference is.

## Causal inference and Inverse Propensity Score

Causal inference is the process of determining causes and effects in a system, from data. This means we want to infer a causal structure from empirical implications. 

A causal model is one where variables are caused by other variables. It is commonly used when studying the effectiveness of a treatment. In that scenario, one wants to discover if taking a certain medicine causes the illness to go away. This is why the terminology is as follows:

- **Outcome**: the state of the observed variable after taking (or not) the treatment. In our case, we could see this as whether the user has clicker or not.
- ******************Treatment******************: the variable we think causes a certain outcome.
- ********************Covariates********************: other variables that can affect the outcome.

Causal inference is important because it allows us to see deeper relationships between variables than more classical statistical methods. It is well known that ************************************correlation does not imply causation.************************************ In the task at hand, we don’t know if a user has clicked an item because it is relevant to their tastes or because it is popular. We could build a model in which only relevance “causes” a click, but that would leave out a possible factor: popularity. 

If we understand the treatment as recommending an item, we want to know if 

# Bibliography

[https://towardsdatascience.com/implementing-causal-inference-a-key-step-towards-agi-de2cde8ea599](https://towardsdatascience.com/implementing-causal-inference-a-key-step-towards-agi-de2cde8ea599)

PETERS, J. y JANZING, D.& S., 2017. Elements of causal inference : foundations and learning algorithms. Cambridge, Massachuestts: The MIT Press. ISBN 0-262-03731-9.

[https://towardsdatascience.com/propensity-score-5c29c480130c](https://towardsdatascience.com/propensity-score-5c29c480130c)
