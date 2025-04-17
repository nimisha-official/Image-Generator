# ⛺ Image Generator with Generative Ai ⛺

 This project demonstrates how to integrate pre-trained diffusion models for image generation into a user-friendly web application. It involves setting up the model pipeline, creating a function to generate images from text prompts, and building a web interface to make the model accessible to users. The use of Gradio simplifies the creation and deployment of the web application, making it easy to share and interact with the model.

### 🟢 Import Required Libraries
``` python 
!pip install diffusers
!pip install transformers
```

### 🟢 Setup Pipline
``` python
from diffusers import StableDiffusionPipeline
import torch

model_id="runwayml/stable-diffusion-v1-5"
sd_pipline=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
sd_pipline=sd_pipline.to('cuda')
```

### 🟢 Make Generater Function
``` python
def Image_Generater(Prompt):
  negative_prompt ="""simple background, duplicate, low quality, lowest quality,
                      bad anatomy, bad proportions, extra digits, 1 wres, username, artist name, error,
                      duplicate, watermark,signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry"""

  return sd_pipline(Prompt,negative_prompt=negative_prompt).images[0]
```

### 🟢 Generate the Image
``` python 
prompt="Cat at restaurant"
gen_img=Image_Generater(prompt)
gen_img.save("./generated_img.jpg")
```

## 📱 Lets Make App

#### import require library
``` python
!pip install gradio==3.48.0
```

#### Code
``` python
import gradio as gr

# Image Generated Function
def Image_Generater(Prompt):
  negative_prompt ="""simple background, duplicate, low quality, lowest quality,
                      bad anatomy, bad proportions, extra digits, 1 wres, username, artist name, error,
                      duplicate, watermark,signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry"""

  return sd_pipline(Prompt,negative_prompt=negative_prompt).images[0]

# make interface
genai_app=gr.Interface(fn=Image_Generater,
                       inputs=[gr.Textbox(label="Write an prompt")],
                       outputs=[gr.Image(label="Generated Image")],
                       title="Image Generater 🖼",
                       description="Generate Image with Stable Diffuser",
                       examples=["cat at restaurant","Humans flying with wings at sky"]
                       )

genai_app.launch(share=True,debug=True)
```

## Screens

#### App screen 1 :
<kbd>![image](https://github.com/Nimisha-Mavar/-Image-Generator/assets/112267753/7f9c4e6a-ca8a-471a-a2bc-215bdb3ae104)</kbd>

#### App screen 2 :
<kbd>![image](https://github.com/Nimisha-Mavar/-Image-Generator/assets/112267753/38fbf868-e75f-40c6-a1bd-b08285de214e)
</kbd>
