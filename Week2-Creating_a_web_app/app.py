import gradio as gr
from fastai.vision.all import *
import gradio as gr


learn = load_learner('brain_ai_model.pkl')


categories = ('Brain', 'Computer')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))
   
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['brain.jpg', 'computer.jpg', 'dunno.jpg']

interface = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
interface.launch(inline=False)
