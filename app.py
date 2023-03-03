from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('summarization', model="s-1-n-t-h/flan-t5", framework='pt',
                     device=device, use_auth_token='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')

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
