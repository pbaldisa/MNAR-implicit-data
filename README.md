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

A **causal model** is a mathematical model representing causal relationships within a system. In other words, variables are *caused* by other variables. It is commonly used when studying the effectiveness of a treatment. In that scenario, one wants to discover if taking a certain medicine causes the illness to go away. This is why the terminology is as follows:

- **Outcome**: the state of the observed variable after taking (or not) the treatment. In our case, we could see this as whether the user has clicker or not.
- **Treatment**: the variable we think causes a certain outcome.
- **Covariates**: other variables that can affect the outcome.

Causal inference is important because it allows us to see deeper relationships between variables than more classical statistical methods. It is well known that **correlation does not imply causation.** In the task at hand, we don’t know if a user has clicked an item because it is relevant to their tastes or because it is popular. We could build a model in which only relevance “causes” a click, but that would leave out a possible factor: popularity. 

To summarise, causal inference aims at finding what causes certain outcomes. In our task at hand, we can use this to be able to differenciate if the user's click was *caused* by an item being relevant.

Now that we have an idea of what causal inference is, what is its relationship with the Propensity Score used by the researchers to tackle the MNAR problem?

The **propensity score** is the probability of treatment assignment conditional on observed baseline characteristics. The propensity score allows one to design and analyze an observational (nonrandomized) study so that it mimics some of the particular characteristics of a randomized controlled trial. In particular, the propensity score is a balancing score: conditional on the propensity score, the distribution of observed baseline covariates will be similar between treated and untreated subjects.[^1]

What this means is that propensity scores are a way to control the covariates by balancing the treated and untreated groups in terms of this score.

The data we have available, is observational data. Thus, it does not control sub-population sizes (i.e. treated and untreated groups may have widely different sizes). Furthermore, the groups may have systematic differences. This is what we want to "control" with the propensity score.

In a randomized controlled trial, the propensity score (probability of treatment) is known and defined by the study. In observational data, however, the true propensity score is generally unknown, but it can be estimated.

# Problem formulation and translation into causal inference terms
Let us introduce some notation. From now on, $u$ refers to a user, and $i$ to an item. Also, let $m$ be the number of users in our dataset, and $n$ the number of items.

Let $Y \in \{0,1\}^{m \times n}$ be a click matrix where each entry $Y_{u, i}$ is a **Bernoulli random variable** representing a click between user $u$ and item $i$. In implicit feedback recommendation, $Y_{u,i} = 1$ indicates positive feedback (i.e. click) and $Y_{u,i} = 0$ is either negative or unlabeled positive feedback (i.e. no click).

In a similar manner, $R \in \{0,1\}^{m \times n}$ is a relevance matrix, where each entry is also a Bernoulli random variable representing the relevance of the user-item pair. If $R_{u,i} = 1$, user $u$ and item $i$ are relevant, otherwise they are irrelevant.

Finally, $O \in \{0,1\}^{m \times n}$ is the exposure matrix. In this case, each entry is a random variable representing whether user $u$ has been exposed to item $i$.

Note that in our case, both relevance and exposure random variables are unobserved. We only possess data on whether a user has or has not clicked an item.

In the study we are analyzing, the researchers make two assumtions for all user-item pairs:
    $$\tag{1} Y_{u,i} = O_{i,i} \cdot R_{u,i} $$
    $$\tag{2}P(Y_{u,i}=1) = P(O_{i,i}=1) \cdot P(R_{u,i}=1)$$

Assumption [(1)](#mjx-eqn-eq1) means that item $i$ is clicked by user $u$ if $i$ has been exposed to $i$ and $i$ is relevant.
Assumption [(2)](#mjx-eqn-eq2) means that the click probability is decomposed into the exposure probability and relevance level. Given this assumption, the exposure probability can take different values among user-item pairs, and it can model the MNAR setting in which the click probability and relevance level are not proportional.

With all this notation in mind, let us formulate the problem in terms of causal inference. Keep in mind, our objective is to deal with the MNAR problem. That is, to differentiate the instances where a user clicked an item because it was relevant, from when they did so because of other variables, such as popularity.

In this scenario, the **treatment** corresponds to being exposed to an item. On the other hand, the outcome is the observable data, i.e. whether the item is clicked. Recalling we defined the propensity score as the probability of receiving treatment, we have: $P(O_{u,i}=1) = P(Y_{u,i}=1 | R_{u_i}=1)$, according to our assumptions.

# Proposed estimator
## Performance metric and ideal loss function
As stated above, the groundbreaking work of the researchers is the use of a different performance metric: relevance level instead of click probability. This metric is more suitable for our task, as it allows us to differentiate between the instances where a user clicked an item because it was relevant, and when they did so because of other variables, such as popularity. Here is the definition of the metric:
    $$\mathcal{R}_{relevance}(\mathcal{\hat{Z}}) = \frac{1}{m} \sum_{u=1}^{m} \sum_{i=1}^{n} P(R_{u,i}=1) \cdot c(\mathcal{\hat{Z}}_{u,i})$$

Where $\mathcal{\hat{Z}}$ is the predicted ranking of item $i$ for user $u$ and the function $c(\mathcal{\hat{Z}}_{u,i})$ characterizes a top-N scoring metric[^2]. In this case, the researchers used the DCG@K metric, which is defined as follows: $c(\mathcal{\hat{Z}}_{u,i}) = \mathbb{1}{(\mathcal{\hat{Z}}_{u,i} \leq K)} / \log(\mathcal{\hat{Z}}_{u,i}+1)$. Note that $P(R_{u,i}=1)$ is the relevance level of the user-item pair.

Basically, this metric is the average relevance level of the user-item pairs weighted by the top-N scoring metric. The goal now is to find a loss function that maximizes this metric. To achieve this, the researchers used the basic pointwise approach. This means that they used a local loss function for each user-item pair, and then averaged the loss across all user-item pairs. If $\mathcal{D}$ is the set of all user-item pairs, the loss function is defined as follows:
    $$\mathcal{L}(\mathcal{\hat{R}}) = \frac{1}{\|\mathcal{D}\|}\sum_{(u,i) \in \mathcal{D}} [P(R_{u,i}=1) \cdot \delta^{(1)}(\mathcal{\hat{R}}_{u,i}) + (1-P(R_{u,i}=1)) \cdot \delta^{(0)}(\mathcal{\hat{R}}_{u,i})]$$

Where $\mathcal{\hat{R}}$ is the predicted relevance level of the user-item pair (which is unobserved), and $\delta^{(1)}$ and $\delta^{(0)}$ are the loss functions for the positive and negative cases, respectively. One could use, for example, a log loss function as $\delta^{(R)}, R \in \{0,1\}$.

Note that a prediction matrix $\mathcal{\hat{R}}$ minimizing the ideal loss function $\mathcal{L}$ is expected to lead to the desired values of the top-N recommendation metric in $\mathcal{R}_{relevance}$. 

//Estimator



# Bibliography
[^1]: 
    [An introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)

[^2]:
    **Top-N scoring metrics** are a family of metrics that measure the performance of a ranking algorithm. They measure how often your predicted class falls in the top N values of your distribution. [Evaluating models using the Top N accuracy metrics](https://medium.com/nanonets/evaluating-models-using-the-top-n-accuracy-metrics-c0355b36f91b)

[https://towardsdatascience.com/implementing-causal-inference-a-key-step-towards-agi-de2cde8ea599](https://towardsdatascience.com/implementing-causal-inference-a-key-step-towards-agi-de2cde8ea599)

PETERS, J. y JANZING, D.& S., 2017. Elements of causal inference : foundations and learning algorithms. Cambridge, Massachuestts: The MIT Press. ISBN 0-262-03731-9.

[https://towardsdatascience.com/propensity-score-5c29c480130c](https://towardsdatascience.com/propensity-score-5c29c480130c)
