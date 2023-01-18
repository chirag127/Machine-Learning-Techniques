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
Genetic Algorithm (GA) is a search heuristic that is inspired by the process of natural selection. It is a optimization technique used to find the best solution to a problem by mimicking the process of natural selection. Here are some notes on Genetic Algorithm with reference to Machine Learning Techniques:

Definition: A Genetic Algorithm is a search heuristic that is inspired by the process of natural selection. It is used to find the best solution to a problem by mimicking the process of natural selection. Genetic Algorithm starts with a population of solutions called chromosomes, and through the process of selection, crossover, and mutation, it evolves the population to find the best solution.

Components: Genetic Algorithm comprises of the following components:

Population: A set of solutions called chromosomes
Fitness Function: A function that measures the quality of a solution
Selection: A process of selecting the best chromosomes from the population
Crossover: A process of combining two chromosomes to create a new one
Mutation: A process of introducing small random changes to a chromosome
Termination: A stopping criterion to decide when to stop the algorithm
GA cycle of reproduction: Genetic Algorithm follows a cyclic process of reproduction where the following steps are repeated to generate new generations:

Selection: The chromosomes are selected based on their fitness values.
Crossover: The selected chromosomes are combined to create new offspring.
Mutation: The new offspring are then mutated to introduce randomness and diversity in the population.
Applications: Genetic Algorithms are widely used in a variety of applications, including optimization of complex functions, scheduling, game playing and trading, image processing, and pattern recognition.

Examples:

A problem of finding the shortest path between two cities can be solved using genetic algorithm.
A problem of finding the global minimum of a function can be solved using genetic algorithm.
In conclusion, Genetic Algorithm is a powerful optimization technique that is inspired by the process of natural selection. It can be used to find the best solution to a problem by mimicking the process of natural selection. The flexibility of GA makes it a popular choice among the researchers and practitioners for solving various optimization problems.



##### Issues in Machine Learning

- Overfitting - The model fits the training data too well. It is not able to generalize well to new data.
- Underfitting - The model is too simple to capture the underlying structure of the data. It is not able to generalize well to new data.
- High variance - The model is too sensitive to small fluctuations in the training data. It is not able to generalize well to new data.
- High bias - The model is too simple to capture the underlying structure of the data. It is not able to generalize well to new data.
- Curse of dimensionality - The number of features is too high. It is not able to generalize well to new data.
- Irrelevant features - The model is too sensitive to irrelevant features. It is not able to generalize well to new data.
- Overlapping classes - The classes are too similar. It is not able to generalize well to new data.
- Small training set - The training set is too small. It is not able to generalize well to new data.
- Small test set - The test set is too small. It is not able to generalize well to new data.
- Noisy data - The data is too noisy. It is not able to generalize well to new data.
- Outliers - The data contains outliers. It is not able to generalize well to new data.
- Imbalanced classes - The classes are imbalanced. It is not able to generalize well to new data.
- Missing values - The data contains missing values. It is not able to generalize well to new data.
In Machine Learning, there are several issues that can arise during the process of building and deploying models. These issues can impact the performance of the model and the overall success of the machine learning project. Here are some notes on issues in Machine Learning:

Overfitting: Overfitting occurs when a model is trained too well on the training data and performs poorly on new, unseen data. This happens when the model is too complex and is able to memorize the training data instead of learning the underlying patterns. Overfitting can be addressed by using techniques such as regularization, early stopping, and ensembling.

Underfitting: Underfitting occurs when a model is not able to capture the underlying patterns in the data and performs poorly on both the training and the test data. This happens when the model is too simple or when the model is not trained for enough number of iterations. Underfitting can be addressed by using more complex models or increasing the number of iterations.

Lack of Data: In some cases, the availability of labeled data can be a limitation. A model may not be able to generalize well to new data if it is trained on a small dataset. This can be addressed by collecting more data, using data augmentation techniques, or using transfer learning.

Data Quality: In some cases, the quality of the data can be a limitation. A model may not be able to generalize well to new data if the data is noisy, inconsistent, or biased. This can be addressed by data cleaning, data preprocessing and data validation.

Bias: Bias refers to the difference between the model's predictions and the true values for certain groups of data. Bias can occur when the training data is not representative of the population or when the model is not designed to handle certain types of data. This can be addressed by using techniques such as oversampling, undersampling, and data augmentation.

Privacy and Security: Machine Learning models may process sensitive data, which can lead to privacy and security issues. For example, a model that is trained on medical data can be used to discriminate against certain groups of people. This can be addressed by using techniques such as differential privacy, homomorphic encryption and federated learning.

In conclusion, issues in Machine Learning can impact the performance of the model and the overall success of the project. It's important to be aware of these issues and to have strategies in place to address them.



##### Data Science vs Machine Learning

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




## Unit 2

### Regression


##### Linear Regression
Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.

Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. Since linear regression shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable.

The linear regression model provides a sloped straight line representing the relationship between the variables. Consider the below image:

Linear Regression in Machine Learning
Mathematically, we can represent a linear regression as:

y= a0+a1x+ ε
Here,

Y= Dependent Variable (Target Variable)
X= Independent Variable (predictor Variable)
a0= intercept of the line (Gives an additional degree of freedom)
a1 = Linear regression coefficient (scale factor to each input value).
ε = random error

The values for x and y variables are training datasets for Linear Regression model representation.

Types of Linear Regression
Linear regression can be further divided into two types of the algorithm:

Simple Linear Regression:
If a single independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Simple Linear Regression.
Multiple Linear regression:
If more than one independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Multiple Linear Regression.
Linear Regression Line
A linear line showing the relationship between the dependent and independent variables is called a regression line. A regression line can show two types of relationship:

Positive Linear Relationship:
If the dependent variable increases on the Y-axis and independent variable increases on X-axis, then such a relationship is termed as a Positive linear relationship.
Linear Regression in Machine Learning
Negative Linear Relationship:
If the dependent variable decreases on the Y-axis and independent variable increases on the X-axis, then such a relationship is called a negative linear relationship.
Linear Regression in Machine Learning
Finding the best fit line:
When working with linear regression, our main goal is to find the best fit line that means the error between predicted values and actual values should be minimized. The best fit line will have the least error.

The different values for weights or the coefficient of lines (a0, a1) gives a different line of regression, so we need to calculate the best values for a0 and a1 to find the best fit line, so to calculate this we use cost function.

Cost function-
The different values for weights or coefficient of lines (a0, a1) gives the different line of regression, and the cost function is used to estimate the values of the coefficient for the best fit line.
Cost function optimizes the regression coefficients or weights. It measures how a linear regression model is performing.
We can use the cost function to find the accuracy of the mapping function, which maps the input variable to the output variable. This mapping function is also known as Hypothesis function.
For Linear Regression, we use the Mean Squared Error (MSE) cost function, which is the average of squared error occurred between the predicted values and actual values. It can be written as:

For the above linear equation, MSE can be calculated as:

Linear Regression in Machine Learning
Where,

N=Total number of observation
Yi = Actual value
(a1xi+a0)= Predicted value.

Residuals: The distance between the actual value and predicted values is called residual. If the observed points are far from the regression line, then the residual will be high, and so cost function will high. If the scatter points are close to the regression line, then the residual will be small and hence the cost function.

Gradient Descent:
Gradient descent is used to minimize the MSE by calculating the gradient of the cost function.
A regression model uses gradient descent to update the coefficients of the line by reducing the cost function.
It is done by a random selection of values of coefficient and then iteratively update the values to reach the minimum cost function.
Model Performance:
The Goodness of fit determines how the line of regression fits the set of observations. The process of finding the best model out of various models is called optimization. It can be achieved by below method:

1. R-squared method:

R-squared is a statistical method that determines the goodness of fit.
It measures the strength of the relationship between the dependent and independent variables on a scale of 0-100%.
The high value of R-square determines the less difference between the predicted values and actual values and hence represents a good model.
It is also called a coefficient of determination, or coefficient of multiple determination for multiple regression.
It can be calculated from the below formula:
Linear Regression in Machine Learning
Assumptions of Linear Regression
Below are some important assumptions of Linear Regression. These are some formal checks while building a Linear Regression model, which ensures to get the best possible result from the given dataset.

Linear relationship between the features and target:
Linear regression assumes the linear relationship between the dependent and independent variables.
Small or no multicollinearity between the features:
Multicollinearity means high-correlation between the independent variables. Due to multicollinearity, it may difficult to find the true relationship between the predictors and target variables. Or we can say, it is difficult to determine which predictor variable is affecting the target variable and which is not. So, the model assumes either little or no multicollinearity between the features or independent variables.
Homoscedasticity Assumption:
Homoscedasticity is a situation when the error term is the same for all the values of independent variables. With homoscedasticity, there should be no clear pattern distribution of data in the scatter plot.
Normal distribution of error terms:
Linear regression assumes that the error term should follow the normal distribution pattern. If error terms are not normally distributed, then confidence intervals will become either too wide or too narrow, which may cause difficulties in finding coefficients.
It can be checked using the q-q plot. If the plot shows a straight line without any deviation, which means the error is normally distributed.
No autocorrelations:
The linear regression model assumes no autocorrelation in error terms. If there will be any correlation in the error term, then it will drastically reduce the accuracy of the model. Autocorrelation usually occurs if there is a dependency between residual errors.


##### Logistic Regression
Logistic Regression is a supervised learning algorithm used for classification problems. It is a statistical method that is used to model the relationship between a set of independent variables and a binary dependent variable. Here are some notes on Logistic Regression:

Definition: Logistic Regression is a statistical method that is used to model the relationship between a set of independent variables and a binary dependent variable. The goal is to find the best set of parameters that maximizes the likelihood of the observed data.

Model: Logistic Regression models the probability of the binary outcome as a function of the input features using the logistic function (sigmoid function). The logistic function produces an output between 0 and 1, which can be interpreted as a probability of the binary outcome. The parameters of the model are learned by maximizing the likelihood of the observed data.

Training: The training process of logistic regression is iterative, it starts with an initial set of parameters, then it uses optimization algorithm like gradient descent to iteratively update the parameters to maximize the likelihood of the observed data.

Evaluation Metrics: The performance of logistic regression models can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

Applications: Logistic Regression is widely used in a variety of applications, including image classification, natural language processing, and bioinformatics.

Examples:

A logistic regression model that is used to predict whether an email is spam or not spam, based on the presence of certain keywords in the email.
A logistic regression model that is used to predict whether a patient has a certain disease based on their symptoms and test results.
In conclusion, Logistic Regression is a widely used algorithm for classification problems. It models the probability of a binary outcome as a function of the input features using the logistic function, and the parameters of the model are learned by maximizing the likelihood of the observed data. It's simple, easy to implement and can provide good results for a large number of problems.

### Bayesian Learning


Bayesian Learning is a statistical learning method based on Bayesian statistics and probability theory. It is used to update the beliefs about the state of the world based on new evidence. Here are some notes on Bayesian Learning:

Definition: Bayesian Learning is a statistical learning method based on Bayesian statistics and probability theory. The goal is to update the beliefs about the state of the world based on new evidence.

Bayes Theorem: Bayesian Learning is based on Bayes' theorem, which states that the probability of a hypothesis given some data (P(H|D)) is proportional to the probability of the data given the hypothesis (P(D|H)) multiplied by the prior probability of the hypothesis (P(H)). Bayes' theorem is often written as P(H|D) = P(D|H) * P(H) / P(D)

Prior and Posterior: In Bayesian Learning, a prior probability distribution is specified for the model parameters, which encodes our prior knowledge or beliefs about the parameters. When new data is observed, the prior is updated to form a posterior probability distribution, which encodes our updated beliefs about the parameters given the data.

Conjugate Prior: To make the calculation of the posterior distribution simple and efficient, it is often useful to use a conjugate prior distribution, which is a distribution that is mathematically related to the likelihood function.

Applications: Bayesian Learning is widely used in a variety of applications, including natural language processing, computer vision, and bioinformatics.

Examples:

A Bayesian Learning model that is used to predict whether a patient has a certain disease based on their symptoms and test results. The prior probability distribution encodes our beliefs about the prevalence of the disease in the population, and the likelihood function encodes the probability of observing the symptoms and test results given the disease.
A Bayesian Learning model that is used to estimate the parameters of a robot's sensor model, by updating the beliefs about the sensor model based on the robot's sensor readings.
In conclusion, Bayesian Learning is a statistical learning method based on Bayesian statistics and probability theory. It is used to update the beliefs about the state of the world based on new evidence. It's powerful and flexible and can be used for a wide range of problems. However, it can be computationally expensive and requires a good understanding of probability theory to implement it correctly.



#### Bayes theorem
Bayes' theorem is a fundamental result in probability theory that relates the conditional probability of an event to the prior probability of the event and the likelihood of the event given some other information. Here are some notes on Bayes' theorem:

Definition: Bayes' theorem states that the conditional probability of an event A given that another event B has occurred is proportional to the prior probability of event A and the likelihood of event B given event A. Mathematically, it can be written as:
P(A|B) = (P(B|A) * P(A)) / P(B)

Intuition: Bayes' theorem is useful for updating our beliefs about the probability of an event given new information. The prior probability is our initial belief before we see the new information, the likelihood is how likely the new information is given our prior belief, and the posterior probability is our updated belief after we see the new information.

Applications: Bayes' theorem has many applications in machine learning, specifically in the field of Bayesian learning. It can be used for classification problems, where the goal is to find the class label of a given input based on the class probabilities and the likelihood of the input given each class. It can also be used for parameter estimation problems, where the goal is to estimate the values of model parameters given some data.

Example:

A weather forecaster states that there is a 30% chance of rain tomorrow, and you observe that it is cloudy today. According to Bayes' theorem, the probability of rain tomorrow given that it is cloudy today can be calculated as:
P(Rain|Cloudy) = (P(Cloudy|Rain) * P(Rain)) / P(Cloudy)
Here, P(Rain|Cloudy) is the posterior probability, P(Cloudy|Rain) is the likelihood, P(Rain) is the prior probability and P(Cloudy) is the normalizing constant.
In conclusion, Bayes' theorem is a fundamental result in probability theory that relates the conditional probability of an event to the prior probability of the event and the likelihood of the event given some other information. It provides a way to update our beliefs about the probability of an event given new information and has many applications in machine learning, particularly in the field of Bayesian learning.


Bayes theorem is a mathematical formula that describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Here are some notes on Bayes theorem:

Definition: Bayes theorem is a mathematical formula that describes the probability of an event, based on prior knowledge of conditions that might be related to the event. It is named after the Reverend Thomas Bayes, who published a paper on the theorem in 1763.

Bayes Theorem: Bayes theorem states that the probability of an event (A) given some evidence (B) is proportional to the probability of the evidence (B) given the event (A) multiplied by the prior probability of the event (A). Bayes theorem is often written as P(A|B) = P(B|A) * P(A) / P(B)

#### Concept learning

Concept learning, also known as concept formation, is a process of learning a general rule or concept from a set of examples. It is a type of unsupervised learning where the goal is to identify the underlying structure or pattern in the data without any prior knowledge of the output. Here are some notes on Concept learning:

Definition: Concept learning is a type of unsupervised learning where the goal is to identify the underlying structure or pattern in the data without any prior knowledge of the output. It is a process of learning a general rule or concept from a set of examples.

Inductive Inference: Concept learning is based on the process of inductive inference. Inductive inference is the process of making generalizations from specific examples. In concept learning, the model uses the examples to infer a general rule or concept that explains the relationship between the input and output.

Concept Learning Algorithms: There are several algorithms used for concept learning, including decision trees, rule-based systems, and nearest-neighbor methods.

Decision Trees: Decision trees are a popular algorithm for concept learning. They use a tree structure to represent the possible concepts and make decisions based on the input.

Rule-based Systems: Rule-based systems use a set of if-then rules to represent the concepts. The rules are learned from the examples and can be used to classify new examples.

Nearest-neighbor Methods: Nearest-neighbor methods are based on the idea that similar examples have similar concepts. The model classifies new examples based on the most similar examples in the training set.

Evaluation Metrics: The performance of concept learning models can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

Applications: Concept learning algorithms are widely used in a variety of applications, including natural language processing, computer vision, speech recognition, and bioinformatics.

Limitations: Concept learning algorithms can be limited by the quality and representativeness of the training data, and may not be able to generalize well to new, unseen data. Additionally, the results of concept learning can be hard to interpret, and the discovered concepts may not be meaningful.

In conclusion, Concept learning is a process of unsupervised learning where the goal is to identify the underlying structure or pattern in the data without any prior knowledge of the output. It is based on the process of inductive inference and several algorithms, such as decision trees, rule-based systems, and nearest-neighbor methods are used to implement it. It's used in several applications and has its own set of limitations and challenges.




#### Bayes Optimal Classifier

The Bayes Optimal Classifier (BOC) is a theoretical classifier that makes the best possible predictions based on the probability of the input data. It is a type of Bayesian classifier that is based on Bayes' theorem, which states that the probability of an event occurring is equal to the prior probability of the event multiplied by the likelihood of the event given certain observations. Here are some notes on the Bayes Optimal Classifier:

Definition: The Bayes Optimal Classifier (BOC) is a theoretical classifier that makes the best possible predictions based on the probability of the input data. It is based on Bayes' theorem and is considered the "gold standard" for classification tasks.

Bayes' Theorem: BOC is based on Bayes' theorem, which states that the probability of an event occurring is equal to the prior probability of the event multiplied by the likelihood of the event given certain observations. In the context of classification, the BOC uses the likelihood of the input data given certain class labels and the prior probability of the class labels to make predictions.

Assumptions: The BOC makes the assumption that the data is generated by a probabilistic process and that the class labels and input features are independent. Additionally, it assumes that the probability distributions of the input data are known.

Evaluation Metrics: The BOC is considered to be the best possible classifier in terms of accuracy, as it makes predictions based on the true underlying probability distributions of the data.

Applications: The BOC is mainly used as a theoretical benchmark for comparing the performance of other classifiers.

Limitations: The BOC is not practical to use in in real-world scenarios, as it requires knowledge of the true underlying probability distributions of the data, which is often not available. Additionally, the BOC can be computationally expensive, as it requires the calculation of the likelihood and prior probabilities for all possible class labels for each input.

Real-world Scenario: In real-world scenarios, the BOC is not practical to use. For example, in the healthcare industry, a doctor may not have the knowledge of the true underlying probability distributions of a patient's medical history and test results to make the best possible predictions about their disease. However, the BOC can be used as a benchmark to evaluate the performance of other classifiers, such as logistic regression or decision trees.
In conclusion, The Bayes Optimal Classifier (BOC) is a theoretical classifier that makes the best possible predictions based on the probability of the input data. It is based on Bayes' theorem, and it's considered the "gold standard" for classification tasks, but it's not practical to use in real-world scenarios as it requires knowledge of the true underlying probability distributions of the data, which is often not available, and it can be computationally expensive. It is mainly used as a theoretical benchmark for comparing the performance of other classifiers.


#### Naive Bayes classifier
Naive Bayes classifier is a probabilistic algorithm based on Bayes theorem for classification tasks. It is a simple and efficient algorithm that makes the naive assumption that all the features are independent from each other. Here are some notes on Naive Bayes classifier:

Definition: Naive Bayes classifier is a probabilistic algorithm based on Bayes theorem for classification tasks. It makes the naive assumption that all the features are independent from each other and uses this assumption to calculate the probability of a class given the input features.

Bayes theorem: Bayes theorem states that the probability of a hypothesis (H) given some evidence (E) is proportional to the probability of the evidence given the hypothesis (P(E|H)) multiplied by the prior probability of the hypothesis (P(H)). In the case of Naive Bayes classifier, the hypothesis is the class, and the evidence is the input features.

Types of Naive Bayes Classifier: There are several types of Naive Bayes classifiers, including Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes.

Gaussian Naive Bayes: This algorithm assumes that the data is normally distributed and calculates the likelihood of the data given the class using the Gaussian distribution.

Multinomial Naive Bayes: This algorithm is used for discrete data, such as text data. It calculates the likelihood of the data given the class using the multinomial distribution.

Bernoulli Naive Bayes: This algorithm is also used for discrete data, such as binary data. It calculates the likelihood of the data given the class using the Bernoulli distribution.

Training: The training process of Naive Bayes classifier is relatively simple. It involves calculating the prior probability of each class and the likelihood of each feature given each class. These probabilities are then used to calculate the posterior probability of each class given the input features. The class with the highest posterior probability is chosen as the final prediction.

Evaluation Metrics: The performance of Naive Bayes classifier can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

Applications: Naive Bayes classifier is widely used in a variety of applications, including text classification, spam filtering, and sentiment analysis.

Limitations: The naive assumption of independence between features can lead to poor performance in cases where the features are highly dependent on each other. Additionally, the performance of Naive Bayes classifier can be sensitive to the presence of irrelevant features and outliers in the data.

In conclusion, Naive Bayes classifier is a simple and efficient algorithm for classification tasks that is based on Bayes theorem. It makes the naive assumption that all the features are independent from each other and uses this assumption to calculate the probability of a class given the input features. It's widely used in several applications, but its performance can be affected by certain factors such as the independence assumption and presence of irrelevant features.






#### Bayesian belief networks


Bayesian belief networks (BBN), also known as Bayesian networks or Belief networks, are probabilistic graphical models that represent the relationships between variables and their probability distributions. They are based on Bayes theorem and are used for reasoning under uncertainty. Here are some notes on Bayesian belief networks:

Definition: Bayesian belief networks (BBN) are probabilistic graphical models that represent the relationships between variables and their probability distributions. They are used for reasoning under uncertainty, and are based on Bayes theorem. BBN represents a set of variables and their conditional dependencies via a directed acyclic graph (DAG). Each node in the graph represents a variable, and each directed edge represents the conditional dependency between two variables.

Conditional Probability: Bayesian belief networks use conditional probability to model the relationships between variables. The probability of a variable given its parents in the graph is represented by a conditional probability distribution. This allows the model to represent the dependencies between variables and how they affect each other.

Inference: Inference in BBN is the process of computing the probability of a variable given the observed values of other variables. This can be done using exact methods such as variable elimination or approximate methods such as Markov Chain Monte Carlo (MCMC).

Learning: Learning in BBN involves estimating the parameters of the conditional probability distributions given a dataset. This can be done using methods such as Maximum Likelihood Estimation (MLE) or Expectation Maximization (EM).

Applications: Bayesian belief networks are widely used in a variety of applications, including medical diagnosis, bioinformatics, natural language processing and computer vision.

Limitations: Bayesian belief networks can be computationally expensive for large datasets or complex models. Additionally, they can be sensitive to missing or noisy data and may not work well for non-linear relationships.

In conclusion, Bayesian belief networks (BBN) are probabilistic graphical models that are used for reasoning under uncertainty. They represent the relationships between variables and their probability distributions via a directed acyclic graph (DAG). They are based on Bayes theorem and can be used for inference and learning. They are widely used in several applications, but they can be computationally expensive for large datasets or complex models.






#### EM algorithm

The Expectation-Maximization (EM) algorithm is an iterative method used for finding maximum likelihood estimates of parameters in models where the data is incomplete or has missing values. It is a widely used technique for estimating parameters in latent variable models, including mixture models and hidden Markov models. Here are some notes on the EM algorithm:

Definition: The Expectation-Maximization (EM) algorithm is an iterative method used for finding maximum likelihood estimates of parameters in models where the data is incomplete or has missing values. It is a widely used technique for estimating parameters in latent variable models, including mixture models and hidden Markov models.

Iterative Process: The EM algorithm consists of two steps: the expectation step (E-step) and the maximization step (M-step). In the E-step, the algorithm estimates the expectation of the complete data log-likelihood function, given the current parameter estimates. In the M-step, the algorithm maximizes the expected complete data log-likelihood function with respect to the parameters, using the estimates from the E-step. This process is repeated until the parameters converge to a maximum likelihood estimate.

Latent Variables: The EM algorithm is particularly useful for models that have latent variables, which are variables that are not directly observed but are inferred from the observed data. Examples of latent variables include the hidden states in a hidden Markov model, or the component membership in a mixture model.

Applications: The EM algorithm is widely used in a variety of applications, including natural language processing, computer vision, bioinformatics, and finance.

Limitations: The EM algorithm can be sensitive to the initialization of the parameters and can converge to a local maximum, rather than the global maximum. Additionally, it can be computationally expensive for large datasets or complex models.

In conclusion, the Expectation-Maximization (EM) algorithm is an iterative method used for finding maximum likelihood estimates of parameters in models where the data is incomplete or has missing values. It is widely used for estimating parameters in latent variable models and is particularly useful for models that have latent variables. It has its own set of limitations and challenges, and its performance can be affected by certain factors such as initialization of the parameters and computational complexity.


### Support Vector Machine


#### Introduction
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression problems. It is a powerful algorithm that can model non-linear decision boundaries by using kernel functions. Here are some notes on Support Vector Machine:

Definition: Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression problems. It is a powerful algorithm that can model non-linear decision boundaries by using kernel functions. The goal of the SVM algorithm is to find the best decision boundary, called the hyperplane, that separates the data into different classes.

Linear SVM: SVM can be used for linear classification problems, where the decision boundary is a straight line or a hyperplane. In this case, the algorithm finds the hyperplane that maximizes the margin, which is the distance between the decision boundary and the closest data points from each class. These closest data points are called support vectors.

Non-linear SVM: SVM can also be used for non-linear classification problems, where the decision boundary is not a straight line or a hyperplane. In this case, the algorithm uses a kernel function to map the input data into a higher-dimensional space where a linear decision boundary can be found. Some commonly used kernel functions are the polynomial kernel, radial basis function (RBF) kernel, and sigmoid kernel.

Training: The training process of SVM involves solving a convex optimization problem to find the optimal hyperplane. The objective is to maximize the margin while keeping the misclassification errors as low as possible.

Evaluation Metrics: The performance of SVM can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

Applications: SVM is widely used in a variety of applications, including image classification, natural language processing, and bioinformatics.

Limitations: SVM can be sensitive to the choice of kernel function, and the model may not converge to the optimal solution in case of noisy or non-separable data. Additionally, SVM can be computationally intensive for large datasets and may not scale well to high-dimensional data.

In conclusion, Support Vector Machine (SVM) is a powerful algorithm for classification and regression problems that can model non-linear decision boundaries using kernel functions. It finds the best decision boundary, called the hyperplane, that separates the data into different classes by maximizing the margin. SVM is widely used in several applications but has certain limitations such as sensitivity to kernel function, computational intensity, and non-convergence in case of noisy data.



#### Types of support vector kernel

Support Vector Machine (SVM) is a supervised learning algorithm that can be used for both classification and regression tasks. One of the key features of SVM is the use of kernel functions, which map the input data into a higher dimensional space in order to find a hyperplane that separates the classes. Here are some notes on the types of support vector kernel:

Linear kernel: The linear kernel is the simplest kernel function and it maps the input data into a linear space. It is defined as the dot product between the input vectors. It is useful when the data is linearly separable.

Polynomial kernel: The polynomial kernel maps the input data into a polynomial space. It is defined as the dot product between the input vectors raised to a power. It can be used when the data is not linearly separable, and it can be useful when the data has a non-linear decision boundary.

Gaussian kernel (RBF): The Gaussian kernel, also known as the radial basis function (RBF) kernel, maps the input data into an infinite-dimensional space. It is defined as the exponential of the negative squared Euclidean distance between the input vectors. It is commonly used in SVM and it is known to work well on a wide range of problems.

Sigmoid kernel: The sigmoid kernel maps the input data into a sigmoid space. It is defined as the tanh of the dot product between the input vectors and a set of parameters. It is less commonly used than the other types of kernels and it is generally used in cases where the data has a non-linear decision boundary.

String kernel: String kernel is a kernel that operates on the sequence of symbols, in this case, the kernel computes the similarity between two strings by counting the number of common substrings. String kernel can be used in text classification and bioinformatics.

In conclusion, the choice of kernel function is an important aspect of SVM algorithm and it can greatly affect the performance of the model. The linear kernel is the simplest and it is useful when the data is linearly separable. The polynomial and Gaussian kernels can be used when the data is not linearly separable and they are commonly used in SVM. The sigmoid kernel is less commonly used and it is generally used in cases where the data has a non-linear decision boundary. String kernel is used in specific applications like text classification and bioinformatics.



##### Linear kernel
The linear kernel is a kernel function used in Support Vector Machine (SVM) that maps the input data into a linear space. It is the simplest kernel function and it is defined as the dot product between the input vectors. Here are some notes on the linear kernel:

Definition: The linear kernel is a kernel function used in Support Vector Machine (SVM) that maps the input data into a linear space. It is defined as the dot product between the input vectors. It is the simplest kernel function used in SVM.

Linear Separability: The linear kernel is useful when the data is linearly separable. In other words, when there exists a linear boundary that can separate the classes. In this case, the linear kernel can find a hyperplane that separates the classes in the original input space.

Computational Efficiency: The linear kernel has the advantage of being computationally efficient. The dot product between two vectors can be computed in O(n) time, where n is the number of features in the input data. This makes it suitable for large datasets or real-time applications.

Limitations: The linear kernel has some limitations when applied to non-linearly separable data. It can't capture complex relationships between the input features and the output classes, and it may not be able to find a good boundary that separates the classes.

Real-world Applications: The linear kernel is widely used in a variety of applications such as text classification, image classification, speech recognition and bioinformatics.

In conclusion, the linear kernel is a kernel function used in SVM, it is defined as the dot product between the input vectors, and it is useful when the data is linearly separable, it has the advantage of being computationally efficient and it is widely used in a variety of applications, but it has limitations when applied to non-linearly separable data.


##### Polynomial kernel

The polynomial kernel is a kernel function used in Support Vector Machine (SVM) algorithm that maps the input data into a higher-dimensional polynomial space. This can be useful when the data is not linearly separable, as it can help to find a hyperplane that separates the classes. Here are some more details on polynomial kernel:

Definition: The polynomial kernel is defined as the dot product between the input vectors raised to a power. It is represented as K(x,x') = (x.x' + c)^d, where x and x' are the input vectors, c is a constant, and d is the degree of the polynomial.

Parameters: The polynomial kernel has three parameters: the degree of the polynomial (d), the constant term (c) and the coefficient term (coef0). The degree of the polynomial (d) controls the complexity of the decision boundary, the constant term (c) controls the trade-off between maximizing the margin and minimizing the classification error, and the coefficient term (coef0) is used to change the bias-variance trade-off.

Non-linear decision boundary: The polynomial kernel can be used when the data is not linearly separable and it can help to find a non-linear decision boundary that separates the classes. The decision boundary created by the polynomial kernel can be more complex than that created by the linear kernel and it can be more suitable for data with a non-linear decision boundary.

Overfitting: If the degree of the polynomial is too high, the polynomial kernel can lead to overfitting. Overfitting occurs when the model is too complex and it fits the noise in the data rather than the underlying pattern. To avoid overfitting, it's important to choose a good value for the degree of the polynomial and also to use cross-validation to choose the best parameters.

Applications: The polynomial kernel is widely used in a variety of applications, such as image classification, text classification, and bioinformatics.

In conclusion, The polynomial kernel is a kernel function that maps the input data into a higher-dimensional polynomial space. It's useful when the data is not linearly separable and can help to find a non-linear decision boundary that separates the classes. The polynomial kernel has several parameters, such as degree of the polynomial, constant term and the coefficient term. It's important to choose a good value for the degree of the polynomial to avoid overfitting and also to use cross-validation to choose the best parameters. It's widely used in several applications.


##### Gaussian kernel

The Gaussian kernel, also known as the radial basis function (RBF) kernel, is a kernel function used in Support Vector Machine (SVM) that maps the input data into an infinite-dimensional space. It is defined as the exponential of the negative squared Euclidean distance between the input vectors. Here are some notes on the Gaussian kernel:

Definition: The Gaussian kernel, also known as the radial basis function (RBF) kernel, is a kernel function used in Support Vector Machine (SVM) that maps the input data into an infinite-dimensional space. It is defined as the exponential of the negative squared Euclidean distance between the input vectors.

Non-linear Separability: The Gaussian kernel is useful when the data is not linearly separable. In other words, when there does not exist a linear boundary that can separate the classes. In this case, the Gaussian kernel can find a boundary that separates the classes in an infinite-dimensional space.

Computational Efficiency: The Gaussian kernel is computationally efficient. Computing the Euclidean distance between two vectors can be done in O(n) time, where n is the number of features in the input data. This makes it suitable for large datasets or real-time applications.

Limitations: One of the main limitations of the Gaussian kernel is its sensitivity to the choice of the kernel's hyperparameter, which determines the width of the Gaussian. If the hyperparameter is set too high, the decision boundary will be too broad and the model will overfit, if it is set too low, the decision boundary will be too narrow and the model will underfit. This can make it difficult to find an optimal value for the hyperparameter. Additionally, Gaussian kernel is computationally expensive for high-dimensional data, as it requires a lot of memory to store the kernel matrix.


Real-world Applications: The Gaussian kernel is widely used in a variety of applications such as text classification, image classification, speech recognition and bioinformatics












#### Hyperplane

A hyperplane is a mathematical concept that is often used in machine learning and specifically in the support vector machine (SVM) algorithm. In simple terms, a hyperplane is a multidimensional line that is used to separate different classes or groups of data points. The goal of the SVM algorithm is to find the best hyperplane that separates the data points into their respective classes with the maximum margin.

In a two-dimensional space, a hyperplane is a line that separates the data points into two classes. However, in higher-dimensional spaces, a hyperplane becomes a hyperplane in n-dimensional space. For example, in a three-dimensional space, a hyperplane is a plane that separates the data points into two classes. In a four-dimensional space, a hyperplane is a hyperplane in four-dimensional space, and so on.

The SVM algorithm uses the concept of a hyperplane to separate the data points into their respective classes. The algorithm begins by finding the hyperplane that separates the data points with the maximum margin. The margin is the distance between the hyperplane and the closest data points from each class. The goal of the SVM algorithm is to find the hyperplane that maximizes the margin between the data points.

The SVM algorithm uses a technique called the kernel trick to find the best hyperplane. The kernel trick is a mathematical technique that allows the SVM algorithm to find the best hyperplane even in higher-dimensional spaces. The kernel trick maps the data points into a higher-dimensional space, where it is easier to find the best hyperplane.

In summary, a hyperplane is a mathematical concept that is used in the SVM algorithm to separate different classes of data points. The goal of the SVM algorithm is to find the best hyperplane that separates the data points into their respective classes with the maximum margin. The SVM algorithm uses a technique called the kernel trick to find the best hyperplane even in higher-dimensional spaces.

Definition: A hyperplane is a subspace that is one dimension lower than the original space. For example, a line is a 1-dimensional hyperplane, a plane is a 2-dimensional hyperplane, and a hyperplane is a 3-dimensional hyperplane. The hyperplane is a linear subspace that separates the input space into two parts.

Separating Hyperplane: In the context of Support Vector Machine (SVM), the hyperplane is a linear subspace that separates the input space into two parts. This is done by maximizing the margin between the hyperplane and the nearest points in each class.

Real-world Applications: The hyperplane is widely used in a variety of applications such as text classification, image classification, speech recognition and bioinformatics.

In conclusion, the hyperplane is a subspace that is one dimension lower than the original space, it is a linear subspace that separates the input space into two parts, and it is widely used in a variety of applications.


#### Properties of SVM


Support Vector Machines (SVMs) are a type of supervised learning algorithm that can be used for classification or regression tasks. Here are some of the key properties of SVMs:

Linear and Non-linear classification: SVMs can be used for both linear and non-linear classification tasks. For linear classification, the decision boundary is a hyperplane. For non-linear classification, the decision boundary is created by transforming the data into a higher-dimensional space where it becomes linearly separable.

Maximal Margin Classifier: The decision boundary that maximizes the margin, which is the distance between the closest data points of different classes, is chosen as the optimal boundary.

Support Vectors: The data points that are closest to the decision boundary are called support vectors. These are the data points that have the most influence on the position of the boundary.

Regularization: SVMs use a regularization parameter, called C, to control the trade-off between maximizing the margin and minimizing the classification error.

Kernel Trick: The kernel trick is a technique used to transform non-linearly separable data into a higher-dimensional space where it becomes linearly separable. This can be done by using different types of kernel functions, such as the polynomial kernel, radial basis function kernel, and sigmoid kernel.

Versatility: SVMs can be used for various type of classification problem, like text classification, image classification, speech recognition, and bioinformatics.

Robustness: SVMs are relatively robust to noise and outliers, as the decision boundary is based on the support vectors, which are the data points closest to the boundary.

Sparse Solution: SVMs tend to find a sparse solution, which means that only a subset of the training data points are used as support vectors. This makes SVMs memory efficient and fast to train.

It's important to note that while SVMs are powerful and versatile models, they can be sensitive to the choice of kernel function, and the choice of the regularization parameter. However, some libraries like scikit-learn have grid search algorithm to help finding the best parameters.


#### Issues in SVM

Support Vector Machines (SVMs) are a popular supervised learning algorithm for classification and regression problems. However, there are a few issues that can arise when using SVMs. Here are some of them:

The choice of kernel: The kernel is a function that maps the input data into a higher-dimensional space, where it is easier to find a linear boundary. Choosing an appropriate kernel can be challenging, as different kernels may perform better or worse depending on the specific problem.

Overfitting: SVMs are prone to overfitting when the number of features is larger than the number of samples. To avoid overfitting, regularization techniques such as the use of soft margins or the addition of a penalty term to the cost function can be used.

Computational complexity: SVMs are computationally expensive, especially when the number of samples is large. This can make training and testing SVMs impractical for large datasets.

High-dimensional problems: When the dimensionality of the problem is high, the number of samples required to achieve good generalization can be prohibitively large. This can make SVMs less effective for high-dimensional problems.

Non-linearly separable problems: SVMs are designed to find linear decision boundaries, and may perform poorly on non-linearly separable problems. To address this, kernel trick can be used to map the data into a higher dimensional space in which a linear boundary can be found.

Handling multi-class problems: SVMs are primarily designed for binary classification problems. To handle multi-class problems, various techniques such as one-vs-all or one-vs-one can be used.

Lack of probabilistic output: SVMs do not provide probabilistic output, which makes it difficult to estimate the uncertainty of the predictions.

Handling missing data: SVMs cannot handle missing data. This can be a problem when working with datasets that have missing values.

To overcome these issues, it is important to carefully tune the parameters of the SVM, and to consider other algorithms such as neural networks, decision trees, and random forests that may be more appropriate for the problem at hand.

## unit 3


### Decision Tree Learning

Decision tree learning is a method for creating a model that can be used to make predictions or decisions. It is a type of supervised learning algorithm that can be used for both classification and regression problems.

A decision tree is a flowchart-like tree structure, where each internal node represents a feature or attribute, each branch represents a decision or rule, and each leaf node represents the outcome or a class label. The topmost node in the decision tree is called the root node, and the bottommost nodes are called leaf nodes.

The process of creating a decision tree involves repeatedly splitting the training data into subsets based on the values of the input features. The goal is to create subsets (or "leaves") that are as pure as possible, meaning that they contain only examples of one class. The splits are chosen so as to maximize a criterion such as information gain or Gini impurity. The resulting tree can then be used to make predictions on new examples by traversing the tree from the root to a leaf node, using the decision rules at each internal node.

There are several algorithms for building decision trees, including:

ID3 (Iterative Dichotomiser 3): This is an early decision tree algorithm that uses information gain to select the best feature to split on at each node.

C4.5: This is an extension of ID3 that can handle both categorical and numerical features. It also uses information gain to select the best feature to split on.

CART (Classification and Regression Trees): This algorithm can be used for both classification and regression problems. It uses Gini impurity as the splitting criterion.

Random Forest: This is an ensemble learning method that generates multiple decision trees and combines their predictions to improve the overall performance.

Decision trees have several advantages over other algorithms such as:

They are easy to understand and interpret.
They can handle both categorical and numerical features.
They can handle missing data.
They can handle large datasets.
However, decision trees also have some disadvantages such as:

They can easily overfit the training data, especially when the tree is deep.
They can be sensitive to small changes in the training data.
They can be biased towards features with many levels.
To overcome these issues, various techniques such as pruning, bagging, and boosting can be used to improve the performance of decision trees.
#### Decision tree learning algorithm
Decision tree learning is a supervised machine learning algorithm that can be used for both classification and regression problems. It creates a model in the form of a tree structure, where each internal node represents a feature or attribute, each branch represents a decision or rule, and each leaf node represents the outcome or a class label.

The basic algorithm for decision tree learning can be broken down into the following steps:

Select the root node: The root node represents the feature that best splits the data into subsets with the highest purity. The purity of a subset is measured by a criterion such as information gain or Gini impurity.

Create internal nodes: For each subset created in the previous step, repeat the process of selecting the feature that best splits the data into subsets with the highest purity.

Create leaf nodes: Once a subset can no longer be split, it becomes a leaf node. The class label of the majority of examples in the subset is assigned to the leaf node.

Prune the tree: To avoid overfitting, the tree can be pruned by removing branches that do not improve the overall performance.

The most popular algorithms for building decision trees include ID3, C4.5, and CART (Classification and Regression Trees).

ID3 (Iterative Dichotomiser 3) is an early decision tree algorithm that uses information gain as the criterion for selecting the best feature to split on at each node. Information gain measures the decrease in entropy (or uncertainty) of the class labels after a feature is used to split the data.

C4.5 is an extension of ID3 that can handle both categorical and numerical features. It also uses information gain as the criterion for selecting the best feature to split on.

CART (Classification and Regression Trees) can be used for both classification and regression problems. It uses Gini impurity as the criterion for selecting the best feature to split on. Gini impurity measures the probability of a randomly chosen example being misclassified if its class label is randomly assigned based on the class distribution of the examples in a subset.

The decision tree algorithm has several advantages, such as ease of interpretation and handling both categorical and numerical features. However, it also has some disadvantages, such as overfitting, sensitivity to small changes in the data and bias towards features with many levels. To overcome these issues, various techniques such as pruning, bagging, and boosting can be used to improve the performance of decision trees.





#### Inductive bias


Inductive bias refers to the assumptions and preconceptions that a machine learning algorithm has about the data it is processing. These biases can affect the algorithm's ability to generalize from the training data to new, unseen data.

Inductive bias can be divided into two main types: positive and negative. Positive inductive bias refers to the assumption that the data follows a certain pattern or structure, which can help the algorithm learn more efficiently. For example, a decision tree algorithm has a positive inductive bias towards hierarchically organized data, as it allows the algorithm to make more efficient splits at each node.

On the other hand, negative inductive bias refers to the assumption that the data does not follow a certain pattern or structure, which can lead to the algorithm missing important information. For example, a linear regression algorithm has a negative inductive bias towards non-linear data, as it assumes that the relationship between the input and output is linear, and may not be able to capture more complex relationships.

It is important to note that while inductive bias can be limiting, it is also necessary for any machine learning algorithm to make predictions. Without any assumptions about the data, the algorithm would have to consider all possible hypotheses, making it computationally infeasible.

The choice of algorithm and its corresponding inductive bias should be chosen based on the nature of the problem and the characteristics of the data. For example, decision tree algorithms are well suited for classification problems with hierarchically organized data, while linear regression algorithms are well suited for problems where the relationship between the input and output is roughly linear.

Overall, Inductive bias is a fundamental aspect of machine learning and understanding it is important for selecting the appropriate algorithm for a given problem and tuning its parameters to achieve the best possible performance.


here are a few examples of inductive bias in machine learning:

Decision Trees: Decision tree algorithms have a positive inductive bias towards hierarchically organized data. This means that they make efficient splits at each node based on the features of the data, allowing them to quickly classify new data points.

Linear Regression: Linear regression algorithms have a negative inductive bias towards non-linear data. This means that they assume that the relationship between the input and output is linear, and may not be able to capture more complex relationships. For example, if we were trying to model the relationship between temperature and ice cream sales, a linear regression algorithm would only consider linear relationships such as a straight line, and not take into account non-linear relationships such as a parabolic shape.

Neural Networks: Neural networks have a positive inductive bias towards discovering complex patterns in data. They are able to learn multiple layers of representations, which can help them to find complex relationships between the input and output. However, as they have a lot of parameters, they can be prone to overfitting.

Naive Bayes: Naive Bayes algorithm have a strong independence assumption, that is, it assumes that all the features of a dataset are independent of each other. This can be a strong assumption for certain datasets, for example, in a dataset with medical records, it will assume that symptoms are independent of the disease, which is not always the case.

k-Nearest Neighbors: k-Nearest Neighbors algorithm has a positive inductive bias towards smooth decision boundaries. This means it will tend to classify new data points based on the majority class of the k nearest training examples. However, this assumption may not be valid for certain types of data, such as data with multiple clusters or non-uniform class distributions.

It is important to note that all machine learning algorithms have some form of inductive bias, and the choice of algorithm and its corresponding bias should be chosen based on the nature of the problem and the characteristics of the data.




##### Inductive inference with decision trees

Inductive inference is the process of using observations to infer general rules or principles. In the context of decision trees, inductive inference is used to construct a decision tree from a set of training data. The process begins by selecting a feature or attribute from the training data and using it as the root node of the tree. The tree is then grown by recursively partitioning the data into subsets based on the values of the selected feature, and creating child nodes for each unique value. This process is repeated for each child node, using a different feature at each level of the tree, until the tree is fully grown. The result is a decision tree that can be used to make predictions about new data by traversing the tree from the root to a leaf node, following the path determined by the values of the features at each level.



Decision trees are a popular method for classification and regression tasks in machine learning. They are a type of predictive model that can be used to map input data (also known as features or attributes) to a target output (such as a class label or a numeric value).

In the case of inductive decision tree learning, the process of constructing a decision tree starts with selecting a feature from the training data as the root node, which is the topmost node of the tree. This feature is chosen based on its ability to best separate or discriminate the target output.

Next, the tree is grown by partitioning the data into subsets based on the values of the selected feature at the root node. Each unique value of the feature corresponds to a child node. This process is repeated for each child node by selecting the next feature that best separates the data at that node. This continues until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples at a leaf node.

As the tree grows, it forms a series of internal nodes and leaf nodes. Internal nodes represent a decision point, where the tree splits the data based on the value of a feature, while leaf nodes represent the final outcome or prediction of the tree. To classify new data, we traverse the tree from the root node to a leaf node by following the path determined by the values of the features at each internal node. The class label or numeric value associated with that leaf node is the prediction made by the tree.

It's worth noting that decision trees are prone to overfitting if the tree is grown too deep. To overcome this problem, techniques such as pruning, which involves removing branches of the tree that do not add much value to the predictions, can be used to reduce the complexity of the tree.
#### Entropy and information theory

##### Entropy Theory

Entropy is a measure of impurity or disorder in a set of data. In the context of decision trees, entropy is used to determine the best feature to split the data on at each step in the tree-building process.

A decision tree is a flowchart-like tree structure, where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition the data into subsets based on the values of the input features.

The entropy of a set S of examples is defined as -∑(p(i) log2 p(i)), where p(i) is the proportion of examples in class i. The entropy is 0 if all examples in S belong to the same class, and it is maximum if the examples in S are evenly split across all classes. In decision tree learning, the goal is to construct a tree that minimizes the entropy of the target class.

At each step in building the tree, the algorithm selects the feature that results in the largest information gain, which is the difference in entropy before and after the split. The feature that results in the highest information gain is chosen as the decision node. This process is repeated on each subset of the data recursively until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

In summary, entropy is a measure of impurity or disorder in a set of data and it is used to determine the best feature to split the data on at each step in the tree-building process. In decision tree learning, the goal is to construct a tree that minimizes the entropy of the target class.

##### Information Theory

Information theory is a branch of mathematics that deals with the study of the transmission, processing, and extraction of information. It was developed by Claude Shannon in the 1940s and is now widely used in fields such as communication engineering, computer science, and statistics.

A decision tree is a graphical representation of a decision-making process. It is used to model a problem and to help find the best solution by breaking it down into smaller, more manageable decisions. Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.

Information theory can be used in the construction of decision trees in several ways. One of the main applications is in the selection of the attribute to test at each internal node. This is typically done by using measures such as information gain or gain ratio, which are based on the concept of entropy, a key concept in information theory. Entropy measures the amount of uncertainty or randomness in a system. In the case of decision trees, it is used to measure the impurity of a set of examples, with a higher entropy indicating a higher level of impurity.

Information gain and gain ratio are used to evaluate the potential of each attribute to reduce the impurity of the set of examples. The attribute with the highest gain or gain ratio is selected to be tested at the next internal node. This process is repeated recursively until the tree is fully grown.

In summary, Information theory provides the mathematical foundation for decision tree learning algorithms by defining the concept of entropy as a measure of impurity, and information gain and gain ratio as measure of the potential of each attribute to reduce the impurity of the set of examples.

##### Information gain and gain ratio

Information gain and gain ratio are two measures of the potential of each attribute to reduce the impurity of the set of examples. The attribute with the highest information gain or gain ratio is selected to be tested at the next internal node. This process is repeated recursively until the tree is fully grown.

Information gain is the difference in entropy before and after a data set is split on an attribute. The attribute with the highest information gain is chosen as the decision node. This process is repeated on each subset of the data recursively until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

Gain ratio is similar to information gain, but it is normalized by the entropy of the attribute. This is done to avoid selecting attributes with a large number of possible values, which would result in a high information gain but a low gain ratio.

In summary, information gain and gain ratio are two measures of the potential of each attribute to reduce the impurity of the set of examples. The attribute with the highest information gain or gain ratio is selected to be tested at the next internal node. This process is repeated recursively until the tree is fully grown.

#### Decision tree learning algorithms

Decision trees are a popular machine learning technique for both classification and regression problems. They are easy to understand and interpret, and can be used to build a predictive model in a few steps. The goal of decision tree learning is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

##### Decision tree learning

Decision tree learning is a supervised machine learning technique for both classification and regression problems. It is used to build a predictive model in a few steps. The goal of decision tree learning is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

A decision tree is a flowchart-like tree structure, where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition the data into subsets based on the values of the input features.

The entropy of a set S of examples is defined as -∑(p(i) log2 p(i)), where p(i) is the proportion of examples in class i. The entropy is 0 if all examples in S belong to the same class, and it is maximum if the examples in S are evenly split across all classes. In decision tree learning, the goal is to construct a tree that minimizes the entropy of the target class.

At each step in building the tree, the algorithm selects the feature that results in the largest information gain, which is the difference in entropy before and after the split. The feature that results in the highest information gain is chosen as the decision node. This process is repeated on each subset of the data recursively until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

In summary, decision tree learning is a supervised machine learning technique for both classification and regression problems. It is used to build a predictive model in a few steps. The goal of decision tree learning is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.


##### Information gain


Information gain is the difference in entropy before and after a data set is split on an attribute. The attribute with the highest information gain is chosen as the decision node. This process is repeated on each subset of the data recursively until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

In summary, information gain is the difference in entropy before and after a data set is split on an attribute. The attribute with the highest information gain is chosen as the decision node. This process is repeated on each subset of the data recursively until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

##### ID-3 Algorithm
The ID3 algorithm starts with the original dataset and selects the attribute with the highest information gain as the root node of the decision tree. Information gain is a measure of the reduction in entropy (or impurity) achieved by partitioning the data based on a particular attribute. The attribute with the highest information gain is considered the most informative attribute, as it provides the most discriminatory power for differentiating between the classes.

Once the root node is selected, the data is partitioned into subsets based on the value of the root attribute. The ID3 algorithm is then recursively applied to each subset to find the next best attribute for each partition. This process continues until a stopping condition is met, such as when all the data belongs to the same class or when there are no more attributes to split on.

The ID3 algorithm is a simple and easy-to-understand algorithm for building decision trees. However, it has some limitations such as overfitting, bias towards attributes with more categories, and handling missing data.

Pruning is one of the way to overcome overfitting issue, ID3 algorithm also can be improved with more advance algorithm such as C4.5 and C5.0 which handle some of its limitation.





ID3 is a popular algorithm for constructing decision trees. It is an iterative algorithm that constructs a decision tree by recursively partitioning the data into subsets based on the values of the input features. The algorithm begins by selecting the feature that best separates the target class, and uses it as the root node of the tree. It then partitions the data into subsets based on the values of the selected feature, and creates child nodes for each unique value. This process is repeated for each child node, using a different feature at each level of the tree, until the tree is fully grown. The result is a decision tree that can be used to make predictions about new data by traversing the tree from the root to a leaf node, following the path determined by the values of the features at each level.

The ID3 algorithm is a greedy algorithm, meaning that it makes the best decision at each step without considering the future consequences of that decision. This can lead to overfitting, where the tree is grown too deep and does not generalize well to new data. To overcome this problem, techniques such as pruning, which involves removing branches of the tree that do not add much value to the predictions, can be used to reduce the complexity of the tree.

In summary, ID3 is a popular algorithm for constructing decision trees. It is an iterative algorithm that constructs a decision tree by recursively partitioning the data into subsets based on the values of the input features. The algorithm begins by selecting the feature that best separates the target class, and uses it as the root node of the tree. It then partitions the data into subsets based on the values of the selected feature, and creates child nodes for each unique value. This process is repeated for each child node, using a different feature at each level of the tree, until the tree is fully grown. The result is a decision tree that can be used to make predictions about new data by traversing the tree from the root to a leaf node, following the path determined by the values of the features at each level.

##### Issues in Decision tree learning

Overfitting: Decision trees can easily overfit the training data if the tree is grown too deep, leading to poor performance on unseen data.

Instability: Small changes in the data can result in a completely different decision tree. This can make the model highly sensitive to minor fluctuations in the training data.

Bias towards attributes with more levels: Decision trees are more likely to split on attributes with more levels or categories, leading to a bias towards those attributes.

Limited use for continuous variables: Decision trees are not well-suited for handling continuous variables and require them to be discretized prior to building the tree.

Unbalanced class problem: Decision tree learning can be biased towards the majority class, making it difficult to handle imbalanced datasets.

Handling missing values: Decision tree algorithms may have difficulty handling missing values, and the results can be affected by how missing values are handled.

Not good for linear relationships: Decision tree is not good for the case where the data has linear relationship.







### Instance-Based Learning

Instance-based learning, also known as memory-based learning, is a type of supervised learning that involves storing instances or examples of the training data in memory and using them as a basis for making predictions. The basic idea behind this approach is to find the most similar instances in the training data to a new input and use them to predict the output.

The main advantage of instance-based learning is that it can handle non-linear and complex relationships between inputs and outputs, as it does not rely on a specific functional form or model structure. This makes it a good choice for problems where the underlying relationship is not well understood or where the data is noisy or non-linear.

There are two main types of instance-based learning algorithms: nearest neighbor and kernel-based.

Nearest neighbor: This algorithm finds the "k" closest instances in the training data to a new input and uses the output of these instances to make a prediction. The most common form of this algorithm is the k-nearest neighbor (k-NN) algorithm, which finds the k closest instances to the new input and assigns the most common output among them as the prediction.

Kernel-based: This algorithm uses a kernel function to find the similarity between instances. The kernel function maps the inputs to a high-dimensional feature space where the similarity is calculated. Common kernel-based algorithms include the support vector machine (SVM) and radial basis function (RBF) network.

Instance-based learning is also computationally efficient and requires very little storage space, as it only needs to store the training instances. However, it can be sensitive to the choice of distance metric and the value of k, and it can be affected by the presence of irrelevant or redundant features in the data.

The main disadvantage of instance-based learning is that it can be slow when making predictions on a large dataset, as it requires searching through all the stored instances to find the most similar ones. Additionally, it is not suitable for online learning, as it requires retraining on the entire dataset whenever new instances are added.






<!--



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
