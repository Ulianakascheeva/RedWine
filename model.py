import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel(model_definition_file='model_definition.yaml')
train_stats = model.train(data_csv='training_dataframe.csv')

# obtain predictions
predictions, test_stats = model.test(data_csv='test_dataframe.csv')

print(predictions)
print('===========================')
print(test_stats)

# closing model
model.close()