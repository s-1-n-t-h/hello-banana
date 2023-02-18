# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline('text2text-generation', model="s-1-n-t-h/bart-summarizer", framework='pt',use_auth_token='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')

if __name__ == "__main__":
    download_model()