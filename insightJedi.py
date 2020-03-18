import numpy as np
import pandas as pd
import category_encoders as ce


def binary_encoder(X, column_number):
    # This method encodes the categorical column
    # specified by column number, into binary values
    # using BinaryEncoder.
    #
    # X   :   input data with nrows and nfeatures
    # column_number: column number which is to be encoded

    if(X.iloc[:, column_number].dtype == 'object'):
        column_data = X.iloc[:, column_number]
        ce_binary = ce.binary.BinaryEncoder()
        encoded_data = ce_binary.fit_transform(column_data)
        return encoded_data

def hashing_encoder(X, column_number):
    # This method encodes the categorical column
    # specified by column number, into fixed length binary values
    # using HashingEncoder.
    #
    # X   :   input data with nrows and nfeatures
    # column_number: column number which is to be encoded

    if(X.iloc[:, column_number].dtype == 'object'):
        column_data = X.iloc[:, column_number]
        ce_hashing = ce.hashing.HashingEncoder(n_components=5)
        encoded_data = ce_hashing.fit_transform(column_data)
        return encoded_data

data = pd.DataFrame({
 'integers' : ['1', '2', '3', '4', '5', '6', '7', '8',
            '9', '10', '11', '12', '13', '14', '15',
               '16', '17', '18', '19', '20']})

encoded_data = binary_encoder(data, 0)
print(encoded_data)
encoded_data = hashing_encoder(data, 0)
print(encoded_data)


data = pd.DataFrame({
 'integers' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})

encoded_data = binary_encoder(data, 0)
print(encoded_data)      # pints None as the input data is integers

