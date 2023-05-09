# Text Classification Model
This is a text classification model that classifies toxic comments using XGBoost.

## Installation
To run the model, you will need to have to install the required packages:
~~~
pip install -r requirements.txt
~~~
## Usage
To run the Streamlit app, run the following command in your terminal:
~~~~
streamlit run streamlit_app.py
~~~~
This will launch a local Streamlit app that can be used to input text and get predictions from the model.

If you would like to train the model yourself, you can use the [text_classification.ipynb](text_classification.ipynb) notebook.

## Files
- [requirements.txt](): A list of Python packages required to run the model.
- [streamlit_app.py](): The Streamlit app code.
- [text_classification.ipynb](): A Jupyter notebook that contains the code used to train the XGBoost model.
- [vectorizer.pkl](): A serialized CountVectorizer object used to transform input text.
- [xgb_classifier.pkl](): A serialized XGBoost model used for classification.

## License
This project is licensed under the MIT License - see the [LICENSE]() file for details.


