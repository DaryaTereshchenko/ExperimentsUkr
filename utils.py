import re
import spacy
from spacymoji import Emoji
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def deEmojify(text):
    nlp = spacy.load('uk_core_news_trf')
    nlp.add_pipe("emoji", first=True)
    doc = nlp(text)
    text = ' '.join(i.text for i in doc if not i._.is_emoji)
    return text


def clean_text(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\s+([?.!,"])', r'\1', text)
    return text


def plot(y_true, y_pred, labels=None, save=False, directory=None, filename=None, title="", cmap=plt.cm.Blues):
    con_mat_df = confusion_matrix(y_true, y_pred)
    con_mat_df = con_mat_df.astype('float') / con_mat_df.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat_df, display_labels=labels)
    disp.plot(cmap=cmap)
    plt.title(title)
    if save:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(directory / filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    clean_text()
    plot()
