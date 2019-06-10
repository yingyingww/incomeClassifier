Relevant data is contained within the data directory. The adult.data and adult.test files contain
the original training and test data, respectively. We split the adult.test file into adult.val
and adult.test2, with adult.val being used as our validation set and adult.test2 being used
as our test set. After this change, we had roughly 2/3 training, 1/6 validation, 1/6 test.


To run the classifiers, use:

python3 classify.py

There are several optional command line arguments. These are detailed below:


usage: classify.py [-h] [--rep] [--csp] [--depth DEPTH] [--plot]
                   [--lr_top LR_TOP] [--lr_bot LR_BOT]
                   [--baseline_attribute BASELINE_ATTRIBUTE] [--depth_plot]

optional arguments:
  -h, --help            show this help message and exit
  --rep                 Conduct reduced error pruning on the decision tree.
                        It's recommended to supply a maximum depth parameter
                        when conducting rpe pruning as this algorithm can take
                        a long time to run otherwise.
  --csp                 Conduct chi square pruning on the decision tree.
  --depth DEPTH         The maximum depth of the decision tree.
  --plot                Plot the accuracy, precision, recall, and f1-score of
                        the classifiers run.
  --lr_top LR_TOP       Get the n largest positively weighted features from
                        the logistic regression model.
  --lr_bot LR_BOT       Get the n largest negatively weighted features from
                        the logistic regression model.
  --baseline_attribute BASELINE_ATTRIBUTE
                        Specify an attribute for the baseline (default is
                        education)
  --depth_plot          Plot maximum depth vs. f-score (for decision tree) and
                        exit



Other notes regarding command line arguments:

Reduced error pruning is very computationally intensive. Running this algorithm with a maximum
depth of 6 took between 5 and 10 minutes to run on my machine. 

Chi-square pruning is something we attempted to implement, but based on results it seems to not be
working as intended. We didn't have the heart to remove it entirely from our submission.

Additionally, chi square pruning and reduced error pruning cannot both be run
at the same time.


Notes on python file structure:

load_data.py - Contains code for loading data from file, and some other useful function used
in multiple classifiers.

decision_tree.py - Contains all code related to the decision tree, including pruning algorithms.

perceptron.py - Contains code related to the perceptron algorithm

classify.py - Contains main function, functions to compute evaluation metrics, and code
relating to the logistic regression model.
