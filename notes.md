# Machine Learning Techniques

## Unit 1

### Introduction

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computer systems to learn from data without being explicitly programmed. The goal of ML is to develop algorithms and models that can automatically identify patterns and make predictions or decisions from data.

In general, there are three main types of ML: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning is the process of training a model on a labeled dataset, where the desired output (label) is provided for each input. Common tasks include classification and regression.
Unsupervised learning is the process of training a model on an unlabeled dataset, where the goal is to find patterns or structure in the data. Common tasks include clustering and dimensionality reduction.
Reinforcement learning is a type of ML where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.
There are various techniques and algorithms used in ML, such as:

Decision Trees: A decision tree is a flowchart-like tree structure, where an internal node represents a feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome.
Support Vector Machines (SVMs): A supervised learning algorithm that can be used for classification or regression tasks.
Neural Networks: A set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns.
Bayesian Networks: A probabilistic graphical model that represents a set of variables and their conditional dependencies via a directed acyclic graph.
Genetic Algorithm: A search heuristic that is inspired by the process of natural selection.
To get started with ML, it's important to have a strong understanding of linear algebra, calculus, and probability. Additionally, many popular platforms and libraries, such as TensorFlow, PyTorch, and scikit-learn, make it easier to implement ML algorithms and models.

#### Learning

1. Learning refers to the process of acquiring new knowledge, skills, or behavior through experience or study. In the context of machine learning, learning refers to the process of training a model or algorithm on a dataset, so that it can make predictions or take actions in new, unseen situations.
2. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learnig.
   1. Supervised learning: The model is trained on a labeled dataset, where the correct output is provided for each input. The goal is to learn the mapping function between inputs and outputs, so that the model can make accurate predictions on new, unseen data.
   2. Unsupervised learning: The model is trained on an unlabeled dataset, where the correct output is not provided for each input. The goal is to discover patterns or relationships in the data, such as clusters or low-dimensional representations
   3. Reinforcement learning: The model learns from feedback from the environment, in the form of rewards or penalties. The goal is to learn a policy that maximizes the cumulative reward over time.
3. The process of learning in machine learning can be broken down into several steps:

Understanding the problem: This includes identifying the inputs, outputs, and constraints of the problem.
Choosing a model: This includes selecting a model architecture and hyperparameters that are appropriate for the problem and dataset.
Choosing a loss function: This includes selecting a function that measures the difference between the model's predictions and the true outputs.
Choosing an optimizer: This includes selecting an algorithm that updates the model's parameters to minimize the loss function.
Choosing a metric: This includes selecting a function that measures the performance of the model on a held-out dataset.
Choosing a dataset: This includes selecting a dataset that is appropriate for the problem and the model.
Machine learning is an iterative process, and the model is typically trained multiple times using different combinations of hyperparameters, architectures, and datasets. The goal is to find the best set of parameters that result in the best performance on the held-out dataset.

Overfitting and underfitting are common issues that arise during the learning process. Overfitting occurs when the model is too complex and memorizes the training data instead of generalizing to new data. Underfitting occurs when the model is too simple and is not able to capture the underlying patterns in the data.

To prevent overfitting, techniques such as regularization and early stopping can be used. To prevent underfitting, techniques such as increasing the model complexity or increasing the size of the dataset can be used.

Please note that the above information is just a brief overview of the topic, and you should consult your course materials and instructors for a more comprehensive understanding of learning in the context of machine learning techniques.

#### Types of Learning

##### Supervised Learning

Supervised Learning is a type of machine learning in which a model or algorithm is trained on a labeled dataset, and then used to make predictions or take actions in new, unseen situations. Here are some notes on Supervised Learning with reference to Machine Learning Techniques:

Definition: Supervised learning is a process where a model learns from labeled training data, which includes input-output pairs, and makes predictions on unseen data. The goal is to learn a function that maps inputs to outputs.

Types of Supervised Learning:

Classification: The goal is to predict a categorical label (e.g. spam or not spam) based on the input data. Common algorithms include logistic regression, decision trees, and support vector machines.

Regression: The goal is to predict a continuous value (e.g. price of a stock) based on the input data. Common algorithms include linear regression and polynomial regression.

Training and Testing: In supervised learning, the training data is used to train the model and the testing data is used to evaluate its performance. The testing data should be independent of the training data to avoid overfitting.

Evaluation Metrics: The performance of a supervised learning model is typically evaluated using metrics such as accuracy, precision, recall, and F1 score (for classification) or mean squared error (for regression).

Applications: Supervised learning algorithms are widely used in a variety of applications, including natural language processing, computer vision, speech recognition, and bioinformatics.

Limitations: Supervised learning algorithms can be limited by the quality and representativeness of the training data, and may not be able to generalize well to new, unseen data. Additionally, they require labeled data to train the model which is not always available and it can be time-consuming and expensive to label data.

###### Detailed information on Supervised Learning

Supervised Learning is a fundamental concept in the field of Machine Learning Techniques, and is widely used in a variety of applications. A comprehensive understanding of Supervised Learning includes not only the basic concepts and definitions but also the underlying algorithms, techniques, and best practices used to implement it.

Definition: Supervised learning is a process where a model learns from labeled training data, which includes input-output pairs, and makes predictions on unseen data. The goal is to learn a function that maps inputs to outputs. The model is trained on a labeled dataset, where each example has an input and an associated output. Once the model is trained, it can be used to make predictions on new, unseen data.

Types of Supervised Learning: There are two main types of supervised learning: classification and regression.

Classification: The goal is to predict a categorical label (e.g. spam or not spam) based on the input data. Common algorithms for classification include logistic regression, decision trees, k-nearest neighbors, and support vector machines.
Regression: The goal is to predict a continuous value (e.g. price of a stock) based on the input data. Common algorithms for regression include linear regression, polynomial regression, and decision tree regression.
Training and Testing: In supervised learning, the training data is used to train the model and the testing data is used to evaluate its performance. The testing data should be independent of the training data to avoid overfitting. A common practice is to split the data into a training set and a test set, where the model is trained on the training set and evaluated on the test set.

Evaluation Metrics: The performance of a supervised learning model is typically evaluated using metrics such as accuracy, precision, recall, and F1 score (for classification) or mean squared error (for regression). These metrics allow us to quantitatively measure how well the model is performing and compare the performance of different models. Other evaluation metrics like ROC-AUC, Log-loss, etc can also be used depending on the type of problem.

Feature Selection and Engineering: Supervised learning models are sensitive to the features of the input data, and the performance of the model can be greatly improved by selecting relevant features and transforming the input data in a way that is more informative. Feature selection techniques like Recursive Feature Elimination, Correlation-based Feature Selection, etc can be used to select the most relevant features and feature engineering techniques like PCA, LDA, etc can be used to transform the input data.

Hyperparameter tuning: Supervised learning models often have a number of parameters that must be set before training. These are called hyperparameters and their optimal values are determined by tuning them on a validation set. Common techniques for hyperparameter tuning include grid search, random search and Bayesian optimization.

Bias-variance trade-off: Supervised learning models can suffer from either high bias or high variance, and it is important to understand the bias-variance trade-off to select the best model. High bias models are simple and tend to underfit the data, while high variance models are complex and tend to overfit the data. Techniques like k-fold cross-validation, regularization, and early stopping can be used to balance bias and variance and improve the performance of the model.

Ensemble Methods: Ensemble methods are used to combine the predictions of multiple models to improve the overall performance. Common ensemble methods include bagging, boosting and stacking. Bagging method like Random Forest and Boosting method like Adaboost are widely used in supervised learning.

Applications: Supervised learning algorithms are widely used in a variety of applications, including natural language processing, computer vision, speech recognition, bioinformatics, financial forecasting, and customer churn prediction.

Limitations: Supervised learning algorithms can be limited by the quality and representativeness of the training data, and may not be able to generalize well to new, unseen data. Additionally, they require labeled data to train the model which is not always available and it can be time-consuming and expensive to label data.

In conclusion, a comprehensive understanding of Supervised Learning with reference to Machine Learning Techniques requires knowledge of the basic concepts, types of learning, underlying algorithms, techniques, and best practices used to implement it, as well as an understanding of its limitations and applications. It's important to keep in mind that this is an active area of research and new techniques and algorithms are being developed all the time.

###### Unsupervised Learning

Unsupervised Learning is a type of machine learning in which a model or algorithm is trained on an unlabeled dataset, and then used to identify patterns or structure in the data. Here are some notes on Unsupervised Learning with reference to Machine Learning Techniques:

Definition: Unsupervised learning is a process where a model learns from unlabeled data, and tries to identify patterns or structure in the data. The goal is to discover hidden structure in the data, rather than making predictions.

Types of Unsupervised Learning: There are several types of unsupervised learning, including clustering, dimensionality reduction, and anomaly detection.

Clustering: The goal is to group similar data points together. Common algorithms include k-means, hierarchical clustering, and DBSCAN.

Dimensionality Reduction: The goal is to reduce the number of features in the data while preserving as much information as possible. Common algorithms include PCA, LLE, and t-SNE.

Anomaly Detection: The goal is to identify data points that do not conform to the general pattern of the data. Common algorithms include One-class SVM and Isolation Forest.

Evaluation Metrics: Evaluating the performance of unsupervised learning models can be challenging, as there is no clear criterion for success. Common metrics include silhouette score, Davies-Bouldin index, and Calinski-Harabasz index for Clustering.

Applications: Unsupervised learning algorithms are widely used in a variety of applications, including image compression, market segmentation, and anomaly detection in financial and healthcare industries.

Limitations: Unsupervised learning algorithms can be limited by the quality and representativeness of the data, and may not be able to identify all the underlying patterns or structure in the data. Additionally, it may not be clear what the output of the model represents and how to interpret the results.

In conclusion, a comprehensive understanding of Unsupervised Learning with reference to Machine Learning Techniques requires knowledge of the basic concepts, types of learning, underlying algorithms, techniques, and best practices used to implement it, as well as an understanding of its limitations and applications. It's important to keep in mind that this is an active area of research and new techniques and algorithms are being developed all the time.

###### Reinforcement Learning Introduction

- Reinforcement Learning is a type of machine learning in which a model or algorithm is trained on an unlabeled dataset, and then used to make predictions or take actions in new, unseen situations. Here are some notes on Reinforcement Learning with reference to Machine Learning Techniques:

- Definition: Reinforcement learning is a process where a model learns from feedback from the environment, in the form of rewards or penalties, and makes predictions or takes actions on unseen data. The goal is to learn a policy that maximizes the cumulative reward over time.

- Types of Reinforcement Learning: There are two main types of reinforcement learning: policy-based and value-based.

- Policy-Based: The goal is to learn a mapping between states and actions, called a policy, which determines the action that should be taken in each state. Common algorithms include policy gradient and Q-learning.

- Value-Based: The goal is to learn a mapping between states and values, called a value function, which estimates the value of being in each state. Common algorithms include policy iteration and value iteration.

- Training and Testing: In reinforcement learning, the training data is used to train the model and the testing data is used to evaluate its performance. The testing data should be independent of the training data to avoid overfitting.

- Evaluation Metrics: The performance of a reinforcement learning model is typically evaluated using metrics such as cumulative reward and average reward per episode.

- Applications:

  - Reinforcement learning algorithms are widely used in a variety of applications, including robotics, video games, and financial trading.
  - Reinforcement learning algorithms are also used in self-driving cars, where the goal is to learn a policy that maximizes the cumulative reward over time, which is the total distance traveled without hitting any obstacles.

- Limitations:
  - Reinforcement learning algorithms can be limited by the quality and representativeness of the training data, and may not be able to generalize well to new, unseen data.
  - It may not be clear what the output of the model represents and how to interpret the results.
  - It may be difficult to determine the optimal policy or value function, and the model may not converge to the optimal solution.
  - It may be difficult to determine the reward function, and the model may not learn the correct policy or value function.

##### Well-defined learning problems

A well-defined learning problem is a crucial aspect of machine learning. It is important to define the problem clearly before attempting to solve it with machine learning techniques. A well-defined learning problem includes a clear statement of the task, input data, output data and performance measure. Here are some notes on well-defined learning problems with reference to Machine Learning Techniques:

Task: The task should be defined clearly and in a specific way. For example, "predicting the price of a stock" is a more specific task than "predicting the stock market."

Input Data: The input data should be clearly defined and understood. It should include the type of data, format, and any preprocessing that needs to be done.

Output Data: The output data should be clearly defined and understood. It should include the type of data, format, and any post-processing that needs to be done.

Performance Measure: The performance measure should be clearly defined and understood. It should include the evaluation metric that will be used to measure the performance of the model.

In Machine Learning Techniques, a well-defined learning problem is a problem where the input, output, and desired behavior of the model are clearly specified. A well-defined learning problem is essential for the successful implementation of a machine learning model. Here are some notes on well-defined learning problems with reference to Machine Learning Techniques:

Definition: A well-defined learning problem is a problem where the input, output, and desired behavior of the model are clearly specified. This includes the type of input, the type of output, and the performance criteria for the model.

Examples:

A supervised learning problem where the input is an image and the output is a label indicating whether the image contains a dog or a cat.
A supervised learning problem where the input is a customer's historical data and the output is a prediction of whether the customer will churn.
A unsupervised learning problem where the input is a set of market transactions and the goal is to find patterns or clusters in the data.
Mnemonics: To remember the importance of well-defined learning problems, one can use the mnemonic "CLEAR"

C stands for "clearly defined inputs and outputs"
L stands for "learning goal is defined"
E stands for "evaluation metric is defined"
A stands for "algorithms are chosen based on the problem"
R stands for "real-world scenario"
Real-world Scenario: In real-world scenarios, a well-defined learning problem is crucial for the successful implementation of machine learning models. For example, in the healthcare industry, a well-defined learning problem would be to predict the likelihood of a patient developing a certain disease based on their medical history and test results. The input would be the patient's medical history and test results, the output would be a probability of the patient developing the disease, and the performance criteria would be the accuracy of the predictions. With a well-defined learning problem, appropriate algorithms can be chosen, and the model can be evaluated using the chosen metric.

In conclusion, a well-defined learning problem is essential for the successful implementation of machine learning models. It allows for the clear specification of inputs, outputs, and desired behavior, which in turn enables the selection of appropriate algorithms and the evaluation of model performance.

##### Designing a Learning System

According to Arthur Samuel “Machine Learning enables a Machine to Automatically learn from Data, Improve performance from an Experience and predict things without explicitly programmed.”

In Simple Words, When we fed the Training Data to Machine Learning Algorithm, this algorithm will produce a mathematical model and with the help of the mathematical model, the machine will make a prediction and take a decision without being explicitly programmed. Also, during training data, the more machine will work with it the more it will get experience and the more efficient result is produced.

![alt_text](https://media.geeksforgeeks.org/wp-content/uploads/20210218081829/MachineLearning.png "image_tooltip")

Example : In Driverless Car, the training data is fed to Algorithm like how to Drive Car in Highway, Busy and Narrow Street with factors like speed limit, parking, stop at signal etc. After that, a Logical and Mathematical model is created on the basis of that and after that, the car will work according to the logical model. Also, the more data the data is fed the more efficient output is produced.

Designing a Learning System in Machine Learning :

According to Tom Mitchell, “A computer program is said to be learning from experience (E), with respect to some task (T). Thus, the performance measure (P) is the performance at task T, which is measured by P, and it improves with experience E.”

Example: In Spam E-Mail detection,

Task, T: To classify mails into Spam or Not Spam.
Performance measure, P: Total percent of mails being correctly classified as being “Spam” or “Not Spam”.
Experience, E: Set of Mails with label “Spam”
Steps for Designing Learning System are:

![img](https://media.geeksforgeeks.org/wp-content/uploads/20210218095354/MLSteps.png "Steps for Designing Learning System are:")

Step 1) Choosing the Training Experience: The very important and first task is to choose the training data or training experience which will be fed to the Machine Learning Algorithm. It is important to note that the data or experience that we fed to the algorithm must have a significant impact on the Success or Failure of the Model. So Training data or experience should be chosen wisely.

Below are the attributes which will impact on Success and Failure of Data:

The training experience will be able to provide direct or indirect feedback regarding choices. For example: While Playing chess the training data will provide feedback to itself like instead of this move if this is chosen the chances of success increases.
Second important attribute is the degree to which the learner will control the sequences of training examples. For example: when training data is fed to the machine then at that time accuracy is very less but when it gains experience while playing again and again with itself or opponent the machine algorithm will get feedback and control the chess game accordingly.
Third important attribute is how it will represent the distribution of examples over which performance will be measured. For example, a Machine learning algorithm will get experience while going through a number of different cases and different examples. Thus, Machine Learning Algorithm will get more and more experience by passing through more and more examples and hence its performance will increase.
Step 2- Choosing target function: The next important step is choosing the target function. It means according to the knowledge fed to the algorithm the machine learning will choose NextMove function which will describe what type of legal moves should be taken. For example : While playing chess with the opponent, when opponent will play then the machine learning algorithm will decide what be the number of possible legal moves taken in order to get success.

Step 3- Choosing Representation for Target function: When the machine algorithm will know all the possible legal moves the next step is to choose the optimized move using any representation i.e. using linear Equations, Hierarchical Graph Representation, Tabular form etc. The NextMove function will move the Target move like out of these move which will provide more success rate. For Example : while playing chess machine have 4 possible moves, so the machine will choose that optimized move which will provide success to it.

Step 4- Choosing Function Approximation Algorithm: An optimized move cannot be chosen just with the training data. The training data had to go through with set of example and through these examples the training data will approximates which steps are chosen and after that machine will provide feedback on it. For Example : When a training data of Playing chess is fed to algorithm so at that time it is not machine algorithm will fail or get success and again from that failure or success it will measure while next move what step should be chosen and what is its success rate.

Step 5- Final Design: The final design is created at last when system goes from number of examples , failures and success , correct and incorrect decision and what will be the next step etc. Example: DeepBlue is an intelligent computer which is ML-based won chess game against the chess expert Garry Kasparov, and it became the first computer which had beaten a human chess expert.

<!--
##### History of Machine Learning

##### Machine Learning Approaches

##### Artificial Neural Network

##### Clustering

##### Reinforcement Learning

##### Decision Tree Learning

##### Bayesian networks

##### Support Vector Machine

##### Genetic Algorithm

##### Issues in Machine Learning

##### Data Science vs Machine Learning


-->

##### History of Machine Learning

It’s all well and good to ask if androids dream of electric sheep, but science fact has evolved to a point where it’s beginning to coincide with science fiction. No, we don’t have autonomous androids struggling with existential crises — yet — but we are getting ever closer to what people tend to call “artificial intelligence.”

Machine Learning is a sub-set of artificial intelligence where computer algorithms are used to autonomously learn from data and information. In machine learning computers don’t have to be explicitly programmed but can change and improve their algorithms by themselves.

Today, machine learning algorithms enable computers to communicate with humans, autonomously drive cars, write and publish sport match reports, and find terrorist suspects. I firmly believe machine learning will severely impact most industries and the jobs within them, which is why every manager should have at least some grasp of what machine learning is and how it is evolving.

In this post I offer a quick trip through time to examine the origins of machine learning as well as the most recent milestones.

1950 — Alan Turing creates the “Turing Test” to determine if a computer has real intelligence. To pass the test, a computer must be able to fool a human into believing it is also human.

1952 — Arthur Samuel wrote the first computer learning program. The program was the game of checkers, and the IBM computer improved at the game the more it played, studying which moves made up winning strategies and incorporating those moves into its program.

1957 — Frank Rosenblatt designed the first neural network for computers (the perceptron), which simulate the thought processes of the human brain.

1967 — The “nearest neighbor” algorithm was written, allowing computers to begin using very basic pattern recognition. This could be used to map a route for traveling salesmen, starting at a random city but ensuring they visit all cities during a short tour.

1979 — Students at Stanford University invent the “Stanford Cart” which can navigate obstacles in a room on its own.

1981 — Gerald Dejong introduces the concept of Explanation Based Learning (EBL), in which a computer analyses training data and creates a general rule it can follow by discarding unimportant data.

Machine Learning (source: Shutterstock)
Machine Learning (source: Shutterstock)
1985 — Terry Sejnowski invents NetTalk, which learns to pronounce words the same way a baby does.

1990s — Work on machine learning shifts from a knowledge-driven approach to a data-driven approach. Scientists begin creating programs for computers to analyze large amounts of data and draw conclusions — or “learn” — from the results.

1997 — IBM’s Deep Blue beats the world champion at chess.

2006 — Geoffrey Hinton coins the term “deep learning” to explain new algorithms that let computers “see” and distinguish objects and text in images and videos.

2010 — The Microsoft Kinect can track 20 human features at a rate of 30 times per second, allowing people to interact with the computer via movements and gestures.

2011 — IBM’s Watson beats its human competitors at Jeopardy.

2011 — Google Brain is developed, and its deep neural network can learn to discover and categorize objects much the way a cat does.

2012 – Google’s X Lab develops a machine learning algorithm that is able to autonomously browse YouTube videos to identify the videos that contain cats.

2014 – Facebook develops DeepFace, a software algorithm that is able to recognize or verify individuals on photos to the same level as humans can.

2015 – Amazon launches its own machine learning platform.

2015 – Microsoft creates the Distributed Machine Learning Toolkit, which enables the efficient distribution of machine learning problems across multiple computers.

2015 – Over 3,000 AI and Robotics researchers, endorsed by Stephen Hawking, Elon Musk and Steve Wozniak (among many others), sign an open letter warning of the danger of autonomous weapons which select and engage targets without human intervention.

2016 – Google’s artificial intelligence algorithm beats a professional player at the Chinese board game Go, which is considered the world’s most complex board game and is many times harder than chess. The AlphaGo algorithm developed by Google DeepMind managed to win five games out of five in the Go competition.

So are we drawing closer to artificial intelligence? Some scientists believe that’s actually the wrong question.

They believe a computer will never “think” in the way that a human brain does, and that comparing the computational analysis and algorithms of a computer to the machinations of the human mind is like comparing apples and oranges.

Regardless, computers’ abilities to see, understand, and interact with the world around them is growing at a remarkable rate. And as the quantities of data we produce continue to grow exponentially, so will our computers’ ability to process and analyze — and learn from — that data grow and expand.

##### Machine Learning Approaches

With the constant advancements in artificial intelligence, the field has become too big to specialize in all together. There are countless problems that we can solve with countless methods. Knowledge of an experienced AI researcher specialized in one field may mostly be useless for another field. Understanding the nature of different machine learning problems is very important. Even though the list of machine learning problems is very long and impossible to explain in a single post, we can group these problems into four different learning approaches:

Supervised Learning;
Unsupervised Learning;
Semi-supervised Learning; and
Reinforcement Learning.
Before we dive into each of these approaches, let’s start with what machine learning is:

What is Machine Learning?
The term “Machine Learning” was first coined in 1959 by Arthur Samuel, an IBM scientist and pioneer in computer gaming and artificial intelligence. Machine learning is considered a sub-discipline under the field of artificial intelligence. It aims to automatically improve the performance of the computer algorithms designed for particular tasks using experience. In a machine learning study, the experience is derived from the training data, which may be defined as the sample data collected on previously recorded observations or live feedbacks. Through this experience, machine learning algorithms can learn and build mathematical models to make predictions and decisions.

The learning process starts by feeding training data (e.g., examples, direct experience, basic instructions) into the model. By using these data, models can find valuable patterns in the data very quickly. These patterns are -then- used to make predictions and decisions on relevant events. The learning may continue even after deployment if the developer builds a suitable machine learning system which allows continuous training.

Four Machine Learning Approaches
Top machine learning approaches are categorized depending on the nature of their feedback mechanism for learning. Most of the machine learning problems may be addressed by adopting one of these approaches. Yet, we may still encounter complex machine learning solutions that do not fit into one of these approaches.

This categorization is essential because it will help you quickly uncover the nature of a problem you may encounter in the future, analyze your resources, and develop a suitable solution.

Let’s start with the supervised learning approach.

Supervised Learning
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.

The supervised learning approach can be adopted when a dataset contains the records of the response variable values (or labels). Depending on the context, this data with labels is usually referred to as “labeled data” and “training data.”

Example 1: When we try to predict a person’s height using his weight, age, and gender, we need the training data that contains people’s weight, age, gender info along with their real heights. This data allows the machine learning algorithm to discover the relationship between height and the other variables. Then, using this knowledge, the model can predict the height of a given person.

Example 2: We can mark e-mails as ‘spam’ or ‘not-spam’ based on the differentiating features of the previously seen spam and not-spam e-mails, such as the lengths of the e-mails and use of particular keywords in the e-mails. Learning from training data continues until the machine learning model achieves a high level of accuracy on the training data.

There are two main supervised learning problems: (i) Classification Problems and (ii) Regression Problems.

Classification Problem
In classification problems, the models learn to classify an observation based on their variable values. During the learning process, the model is exposed to a lot of observations with their labels. For example, after seeing thousands of customers with their shopping habits and gender information, a model may successfully predict the gender of a new customer based on his/her shopping habits. Binary classification is the term used for grouping under two labels, such as male and female. Another binary classification example might be predicting whether the animal in a picture is a ‘cat’ or ‘not cat,’ as shown in Figure 2–4.

On the other hand, multilabel classification is used when there are more than two labels. Identifying and predicting handwritten letters and numbers on an image would be an example of multilabel classification.

Regression Problems
In regression problems, the goal is to calculate a value by taking advantage of the relationship between the other variables (i.e., independent variables, explanatory variables, or features) and the target variable (i.e., dependent variable, response variable, or label). The strength of the relationship between our target variable and the other variables is a critical determinant of the prediction value. Predicting how much a customer would spend based on its historical data is a regression problem.

Unsupervised Learning
Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision.

Unsupervised learning is a learning approach used in ML algorithms to draw inferences from datasets, which do not contain labels.

There are two main unsupervised learning problems: (i) Clustering and (ii) Dimensionality Reduction.

Clustering
Unsupervised learning is mainly used in clustering analysis.

Clustering analysis is a grouping effort in which the members of a group (i.e., a cluster) are more similar to each other than the members of the other clusters.

There are many different clustering methods available. They usually utilize a type of similarity measure based on selected metrics such as Euclidean or probabilistic distance. Bioinformatic sequence analysis, genetic clustering, pattern mining, and object recognition are some of the clustering problems that may be tackled with the unsupervised learning approach.

Dimensionality Reduction
Another use case of unsupervised learning is dimensionality reduction. Dimensionality is equivalent to the number of features used in a dataset. In some datasets, you may find hundreds of potential features stored in individual columns. In most of these datasets, several of these columns are highly correlated. Therefore, we should either select the best ones, i.e., feature selection, or extract new features combining the existing ones, i.e., feature extraction. This is where unsupervised learning comes into play. Dimensionality reduction methods help us create neater and cleaner models that are free of noise and unnecessary features.

Semi-Supervised Learning
Semi-supervised learning is a machine learning approach that combines the characteristics of supervised learning and unsupervised learning. A semi-supervised learning approach is particularly useful when we have a small amount of labeled data with a large amount of unlabeled data available for training. Supervised learning characteristics help take advantage of the small amount of label data. In contrast, unsupervised learning characteristics are useful to take advantage of a large amount of unlabeled data.

Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training.

Well, you might think that if there are useful real-life applications for semi-supervised learning. Although supervised learning is a powerful learning approach, labeling data -to be used in supervised learning- is a costly and time-consuming process. On the other hand, a sizeable data volume can also be beneficial even though they are not labeled. So, in real life, semi-supervised learning may shine out as the most suitable and the most fruitful ML approach if done correctly.

In semi-supervised learning, we usually start by clustering the unlabeled data. Then, we use the labeled data to label the clustered unlabeled data. Finally, a significant amount of now-labeled data is used to train machine learning models. Semi-supervised learning models can be very powerful since they can take advantage of a high volume of data.

Semi-supervised learning models are usually a combination of transformed and adjusted versions of the existing machine learning algorithms used in supervised and unsupervised learning. This approach is successfully used in areas like speech analysis, content classification, and protein sequence classification. The similarity of these fields is that they offer abundant unlabeled data and only a small amount of labeled data.

Reinforcement Learning
Reinforcement learning is one of the primary approaches to machine learning concerned with finding optimal agent actions that maximize the reward within a particular environment. The agent learns to perfect its actions to gain the highest possible cumulative reward.

Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

There are four main elements in reinforcement learning:

Agent: The trainable program which exercises the tasks assigned to it
Environment: The real or virtual universe where the agent completes its tasks.
Action: A move of the agent which results in a change of status in the environment
Reward: A negative or positive remuneration based on the action.
Reinforcement learning may be used in both the real world as well as in the virtual world:

Example 1: You may create an evolving ad placement system deciding how many ads to place on a website based on the ad revenue generated in different setups. The ad placement system would be an excellent example of real-world applications.

Example 2: On the other hand, you can train an agent in a video game with reinforcement learning to compete against other players, usually referred to as bots.

Example 3: Finally, virtual and real robots training in terms of their movements are done with the reinforcement learning approach.

Some of the popular reinforcement learning models may be listed as follows:

Q-Learning,
State-Action-Reward-State-Action (SARSA),
Deep Q Network (DQN),
Deep Deterministic Policy Gradient (DDPG),
One of the disadvantages of the popular deep learning frameworks is that they lack comprehensive module support for reinforcement learning, and TensorFlow and PyTorch are no exception. Deep reinforcement learning can only be done with extension libraries built on top of existing deep learning libraries such as Keras-RL, TF.Agents, and Tensorforce or dedicated reinforcement learning libraries such as Open AI Baselines and Stable Baselines.

Now that we covered all four approaches, here is a summary visual to make basic comparison among different ML approaches:

Final Notes
The field of AI is expanding very quickly and becoming a major research field. As the field expands, sub-fields and sub-subfields of AI have started to appear. Although we cannot master the entire field, we can at least be informed about the major learning approach.

The purpose of this post was to make you acquainted with these four machine learning approaches. In the upcoming post, we will cover other AI essentials.

##### Artificial Neural Network

Artificial neural networks (ANNs), usually simply called neural networks (NNs) or neural nets, are computing systems inspired by the biological neural networks that constitute animal brains.

An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold.

Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.

The original goal of neural networks was to model the brain, but they have also been used for statistical modeling and data mining. Neural networks are used in applications such as speech recognition, image recognition, medical diagnosis, and natural language processing. Neural networks are also used in reinforcement learning, a form of machine learning that allows software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance.

##### Artificial Neuron

Artificial Neuron is a mathematical model of a biological neuron. It is a mathematical function that takes a set of inputs, performs a calculation on them, and produces a single output. The output of the artificial neuron is a function of the inputs and the weights associated with the inputs. The artificial neuron is a basic unit of a neural network. It is a mathematical function that takes a set of inputs, performs a calculation on them, and produces a single output. The output of the artificial neuron is a function of the inputs and the weights associated with the inputs. The artificial neuron is a basic unit of a neural network.

##### Clustering

- Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). Clustering is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.

- Clustering is a method of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of clustering is to identify the inherent groupings in the data, such as grouping customers by purchasing behavior. Clustering is also used for exploratory data analysis to find hidden patterns or grouping in data.

- In the field of machine learning, clustering is one of the most popular unsupervised learning methods. Clustering is used for exploratory data analysis to find hidden patterns or grouping in data. Clustering is also used for customer segmentation, recommender systems, image segmentation, semi-supervised learning, and dimensionality reduction.

- The main goal of clustering is to group the data into distinct groups, such that the data points in the same group are similar to each other and dissimilar to the data points in other groups. There are several clustering algorithms, such as K-means, hierarchical clustering, DBSCAN, and Gaussian mixture models. The K-means algorithm is one of the most popular clustering algorithms and is used for a wide range of applications.

- Types of clustering

###### Partitioning Clustering

- Partitioning Clustering is a type of clustering that divides the data into non-hierarchical groups. It is also known as the centroid-based method. The most common example of partitioning clustering is the K-Means Clustering algorithm.
- In this type, the dataset is divided into a set of k groups, where K is used to define the number of pre-defined groups. The cluster center is created in such a way that the distance between the data points of one cluster is minimum as compared to another cluster centroid.

###### Density-Based Clustering

- The density-based clustering method connects the highly-dense areas into clusters, and the arbitrarily shaped distributions are formed as long as the dense region can be connected. This algorithm does it by identifying different clusters in the dataset and connects the areas of high densities into clusters. The dense areas in data space are divided from each other by sparser areas.
- These algorithms can face difficulty in clustering the data points if the dataset has varying densities and high dimensions.

###### Hierarchical Clustering

- Hierarchical clustering can be used as an alternative for the partitioned clustering as there is no requirement of pre-specifying the number of clusters to be created. In this technique, the dataset is divided into clusters to create a tree-like structure, which is also called a dendrogram. The observations or any number of clusters can be selected by cutting the tree at the correct level. The most common example of this method is the Agglomerative Hierarchical algorithm.

###### Fuzzy Clustering

- Fuzzy clustering is a type of soft method in which a data object may belong to more than one group or cluster. Each dataset has a set of membership coefficients, which depend on the degree of membership to be in a cluster. Fuzzy C-means algorithm is the example of this type of clustering; it is sometimes also known as the Fuzzy k-means algorithm.

###### Clustering Algorithms

- The Clustering algorithms can be divided based on their models that are explained above. There are different types of clustering algorithms published, but only a few are commonly used. The clustering algorithm is based on the kind of data that we are using. Such as, some algorithms need to guess the number of clusters in the given dataset, whereas some are required to find the minimum distance between the observation of the dataset.

Here we are discussing mainly popular Clustering algorithms that are widely used in machine learning:

K-Means algorithm: The k-means algorithm is one of the most popular clustering algorithms. It classifies the dataset by dividing the samples into different clusters of equal variances. The number of clusters must be specified in this algorithm. It is fast with fewer computations required, with the linear complexity of O(n).
Mean-shift algorithm: Mean-shift algorithm tries to find the dense areas in the smooth density of data points. It is an example of a centroid-based model, that works on updating the candidates for centroid to be the center of the points within a given region.
DBSCAN Algorithm: It stands for Density-Based Spatial Clustering of Applications with Noise. It is an example of a density-based model similar to the mean-shift, but with some remarkable advantages. In this algorithm, the areas of high density are separated by the areas of low density. Because of this, the clusters can be found in any arbitrary shape.
Expectation-Maximization Clustering using GMM: This algorithm can be used as an alternative for the k-means algorithm or for those cases where K-means can be failed. In GMM, it is assumed that the data points are Gaussian distributed.
Agglomerative Hierarchical algorithm: The Agglomerative hierarchical algorithm performs the bottom-up hierarchical clustering. In this, each data point is treated as a single cluster at the outset and then successively merged. The cluster hierarchy can be represented as a tree-structure.
Affinity Propagation: It is different from other clustering algorithms as it does not require to specify the number of clusters. In this, each data point sends a message between the pair of data points until convergence. It has O(N2T) time complexity, which is the main drawback of this algorithm.

Applications of Clustering

Below are some commonly known applications of clustering technique in Machine Learning:

In Identification of Cancer Cells: The clustering algorithms are widely used for the identification of cancerous cells. It divides the cancerous and non-cancerous data sets into different groups.
In Search Engines: Search engines also work on the clustering technique. The search result appears based on the closest object to the search query. It does it by grouping similar data objects in one group that is far from the other dissimilar objects. The accurate result of a query depends on the quality of the clustering algorithm used.
Customer Segmentation: It is used in market research to segment the customers based on their choice and preferences.
In Biology: It is used in the biology stream to classify different species of plants and animals using the image recognition technique.
In Land Use: The clustering technique is used in identifying the area of similar lands use in the GIS database. This can be very useful to find that for what purpose the particular land should be used, that means for which purpose it is more suitable.


#### Reinforcement Learning

Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

Reinforcement learning differs from supervised learning in not needing labelled input/output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).

The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context use dynamic programming techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.


##### Decision Tree Learning

Decision tree learning is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves).

Decision tree learning is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves).

###### Decision tree learning algorithm

The decision tree learning algorithm is a supervised learning algorithm that can be used for both classification and regression problems. It is a non-parametric supervised learning method used for classification and regression. It is a non-parametric supervised learning method used for classification and regression. It is a non-parametric supervised learning method used for classification and regression. It is a non-parametric supervised learning method used for classification and regression.

###### Decision tree learning advantages

- Decision trees are easy to understand and interpret. People are able to understand decision tree models after a brief explanation.
- Decision trees require little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
- Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
- Able to handle multi-output problems.
- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


###### Decision tree learning disadvantages

- Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
###### Decision tree learning applications

- Decision trees are used in data mining for the purpose of classification and regression. They are one of the most popular tools and are used in a wide variety of applications, including medicine, manufacturing, and finance.


###### Bayesian networks

- Bayesian networks are a type of probabilistic graphical model that represent a set of variables and their conditional dependencies via a directed acyclic graph (DAG). Bayesian networks are a type of probabilistic graphical model that represent a set of variables and their conditional dependencies via a directed acyclic graph (DAG). Bayesian networks are a type of probabilistic graphical model that represent a set of variables and their conditional dependencies via a directed acyclic graph (DAG). Bayesian networks are a type of probabilistic graphical model that represent a set of variables and their conditional dependencies via a directed acyclic graph (DAG).




##### Support Vector Machine


Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. SVMs are a set of related supervised learning methods used for classification, regression and outliers detection. SVMs are a set of related supervised learning methods used for classification, regression and outliers detection. SVMs are a set of related supervised learning methods used for classification, regression and outliers detection. SVMs are a set of related supervised learning methods used for classification, regression and outliers detection.

###### Support vector machine advantages

- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

###### Support vector machine disadvantages

- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

###### Support vector machine applications

- Face detection
- Text and hypertext categorization
- Classification of images
- Bioinformatics
- Protein fold and remote homology detection
- Handwriting recognition
- Generalized predictive control

###### Support vector machine kernel functions

- Linear
- Polynomial
- Radial basis function (RBF)
- Sigmoid
- Precomputed

###### Support vector machine kernel types

- Linear: \[K(x_i, x_j) = x_i^T x_j\]
- Polynomial: \[K(x_i, x_j) = \left( \gamma x_i^T x_j + r \right)^d\]
- Radial basis function (RBF): \[K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)\]
- Sigmoid: \[K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)\]

###### Support vector machine kernel parameters

- \gamma = \frac{1}{2\sigma^2} for RBF, polynomial and sigmoid.
- r is the offset used in polynomial and sigmoid kernels.
- d is the degree of the polynomial kernel.

###### Support vector machine regularization parameters

- C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
- \epsilon is the epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance \epsilon from the actual value.

###### Support vector machine kernel cache size

- The kernel cache size is given by the cache_size parameter. The kernel cache stores the results of the kernel evaluations for the samples.


##### Genetic Algorithm



<!--



##### Issues in Machine Learning

##### Data Science vs Machine Learning

##### Regression

##### Linear Regression

##### Logistic Regression

##### Bayesian Learning

##### Bayes theorem

##### Concept learning

##### Bayes Optimal Classifier

##### Naive Bayes classifier

##### Bayesian belief networks

##### EM algorithm

##### Support Vector Machine

##### Introduction

##### Types of support vector kernel

##### Linear kernel

##### Polynomial kernel

##### Gaussian kernel

##### Hyperplane

##### Properties of SVM

##### Issues in SVM

##### Decision Tree Learning

##### Decision tree learning algorithm

##### Inductive bias

##### Inductive inference with decision trees

##### Entropy and information theory

##### Information gain

##### ID-3 Algorithm

##### Issues in Decision tree learning

##### Instance-Based Learning

##### k-Nearest Neighbour Learning

##### Locally Weighted Regression

##### Radial basis function networks

##### Case-based learning

##### Artificial Neural Networks

##### Perceptrons

##### Multilayer perceptron

##### Gradient descent and the Delta rule

##### Multilayer networks

##### Derivation of Backpropagation Algorithm

##### Generalization

##### Unsupervised Learning

##### SOM Algorithm and its variant

##### Deep Learning

##### Introduction

##### Concept of convolutional neural network

##### Types of layers

##### Concept of Convolution (1D and 2D) layers

##### Training of network

##### Case study of CNN for eg on Diabetic Retinopathy

##### Building a smart speaker

##### Self-deriving car etc

##### Reinforcement Learning

##### Introduction to Reinforcement Learning

##### Learning Task

##### Example of Reinforcement Learning in Practice

##### Learning Models for Reinforcement

##### Application of Reinforcement Learning

##### Introduction to Deep Q Learning

##### Genetic Algorithms

##### Introduction

##### Components

##### GA cycle of reproduction

##### Crossover

##### Mutation

##### Genetic Programming

##### Models of Evolution and Learning

##### Applications -->
