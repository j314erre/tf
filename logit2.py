import argparse
import logging
import statsmodels.api as sm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



logger = logging.getLogger("logit2")


################################ MAIN #######################

def main():
    
    # define args
    parser = argparse.ArgumentParser(description='Predict KPI given text using logistic regression')
    parser.add_argument("--input", help="Input tab-delimited data file")
    parser.add_argument("--text_column", help="Column containing text (default is 1st column)", type=int, default=0)
    parser.add_argument("--kpi_column", help="Column containing KPI value (default is 2nd column)", type=int, default=1)
    parser.add_argument("--max_features", help="Max number of word features in model", type=int, default=100)

        
    # set up logger
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # parse command line
    args = parser.parse_args()

    kpi_column = args.kpi_column
    text_column = args.text_column
    
    # read data file
    train = pd.read_csv(args.input, header=None, delimiter="\t", quoting=3)   
     

    logger.info(train.shape)
    
    # build vocab
    vectorizer = CountVectorizer(min_df=1,binary=True,max_features=args.max_features)
    
    docs = []
    for line in train[text_column]:
        docs.append(line)

                
    # select data features
    train_data_features = vectorizer.fit_transform(docs)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features_array = train_data_features.toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    logger.info(vocab)
#     for i in range(len(Y_test)):
#         print Y_test[i], '\t', predict(X_test[i], w)

    # create data frames for pandas
    train_dataframe = pd.DataFrame(train_data_features_array, columns=vocab)
    kpi_dataframe = pd.DataFrame(data={"kpi":train[kpi_column]} )

    # set intercept
    train_dataframe['intercept'] = 1.0
    
    logger.info(train_dataframe.head())
    
    train_cols = train_dataframe.columns[0:]
    
    # use Logit model
    logit = sm.Logit(kpi_dataframe['kpi'], train_dataframe[train_cols], missing='drop')
 
    logger.info("FITTING MODEL...")
    
    # fit the model
    result = logit.fit()

    logger.info( "PRINT SUMMARY...")

    # output results
    logger.info(result.summary())
    
if __name__ == "__main__": 
    main()




