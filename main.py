import os
import numpy as np
import uvicorn
import traceback
import tensorflow as tf

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('hate_speech_model.h5')
app = FastAPI()

@app.get("/")
def index():
    return "Hate Speech"

# If your model need text input use this endpoint!
class RequestText(BaseModel):
    text:str

@app.post("/predict_text")

def predict_text(req: RequestText, response: Response):

    try:
        text = req.text
        text = np.array(tokenizer.texts_to_sequences(text.tolist()))
        text = pad_sequences(text, padding='post', maxlen=maxlen)
        
        print("Uploaded text:", text)
        
        # Step 1: (Optional) Do your text preprocessing
        
        # Step 2: Prepare your data to your model
        
        # Step 3: Predict the data
	
	      
        result = model.predict(text)
        cutoff=0.86
        test_data['pred_sentiment']= result
        test_data['pred_sentiment'] = np.where((test_data.pred_sentiment >= cutoff),1,test_data.pred_sentiment)
        test_data['pred_sentiment'] = np.where((test_data.pred_sentiment < cutoff),0,test_data.pred_sentiment)
	
        
        # Step 4: Change the result your determined API output
        
        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

#port = os.environ.get("PORT", 8000)
#print(f"Listening to http://0.0.0.0:{port}")
#uvicorn.run(app, host='0.0.0.0',port=port)