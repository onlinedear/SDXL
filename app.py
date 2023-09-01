import os  # 导入操作系统模块
os.system("pip install openxlab")  # 使用系统命令安装指定的Python包
os.system('pip install invisible_watermark safetensors')  # 使用系统命令安装指定的Python包
os.system('pip install diffusers==0.18.0')  # 使用系统命令安装指定版本的Python包
import cv2  # 导入OpenCV库
import torch  # 导入PyTorch库
import gradio as gr  # 导入Gradio库
import numpy as np  # 导入NumPy库
from modelscope.utils.constant import Tasks  # 从modelscope.utils.constant模块导入Tasks枚举
from transformers import pipeline

# 定义提示词的字典
prompt_dict={
    "None": "{prompt}",
    "Enhance": "breathtaking {prompt} . award-winning, professional, highly detailed",
    "Anime": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
    "Photographic": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "Digital Art": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    "Comic Book": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "Fantasy Art": "ethereal fantasy concept art of {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    "Analog Film": "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
    "Neon Punk": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
    "Isometric": "isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
    "Low Poly": "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
    "Origami": "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
    "Line Art": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
    "Craft Clay": "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
    "Cinematic": "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
    "3D Model": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    "Pixel Art": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    "Texture": "texture {prompt} top down close-up"
    }
# 定义负向提示词的字典
negative_prompt_dict={
    "None": "{negative_prompt}",
    "Enhance": "{negative_prompt} ugly, deformed, noisy, blurry, distorted, grainy",
    "Anime": "{negative_prompt} photo, deformed, black and white, realism, disfigured, low contrast",
    "Photographic": "{negative_prompt} drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    "Digital Art": "{negative_prompt} photo, photorealistic, realism, ugly",
    "Comic Book": "{negative_prompt} photograph, deformed, glitch, noisy, realistic, stock photo",
    "Fantasy Art": "{negative_prompt} photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    "Analog Film": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Neon Punk": "{negative_prompt} painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "Isometric": "{negative_prompt} deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
    "Low Poly": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Origami": "{negative_prompt} noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Line Art": "{negative_prompt} anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    "Craft Clay": "{negative_prompt} sloppy, messy, grainy, highly detailed, ultra textured, photo",
    "Cinematic": "{negative_prompt} anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    "3D Model": "{negative_prompt} ugly, deformed, noisy, low poly, blurry, painting",
    "Pixel Art": "{negative_prompt} sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    "Texture": "{negative_prompt} ugly, deformed, noisy, blurry"
    }

# 定义清除函数，用于重置输入参数
def clear_fn(value):
    return "", "", "None", 1024, 1024, 10, 50, None

# 定义将多张图片拼接成一张图片的函数
def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset+img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image

# 创建图像合成的管道，引用OpenXlab SDXL模型
pipe = pipeline(task=Tasks.text_to_image_synthesis, 
            model='camenduru/sdxl-base-1.0',
            use_safetensors=True,
            model_revision='v1.0.0')

# 定义显示图像合成结果的函数
def display_pipeline(prompt: str,
                      negative_prompt: str, 
                      style: str = 'None',
                      height: int = 1024, 
                      width: int = 1024, 
                      scale: float = 10, 
                      steps: int = 50, 
                      seed: int = 0):
    if not prompt:
       raise gr.Error('没有输入任何提示词')
    print(prompt_dict[style])
    prompt = prompt_dict[style].format(prompt=prompt)
    negative_prompt = negative_prompt_dict[style].format(negative_prompt=negative_prompt)

    generator = torch.Generator(device='cuda').manual_seed(seed)
    output = pipe({'text': prompt, 
                    'negative_prompt': negative_prompt,
                    'num_inference_steps': steps,
                    'guidance_scale': scale,
                    'height': height,
                    'width': width,
                    'generator': generator
                    })
    result = output['output_imgs'][0]

    image_path = './lora_result.png'
    cv2.imwrite(image_path, result)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image
   
# 创建Gradio界面布局
with gr.Blocks() as demo:
    # 创建第一行
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label='提示词',lines=3)  # 创建一个文本框用于输入提示词
            negative_prompt = gr.Textbox(label='负向提示词',lines=3) # 创建一个文本框用于输入负向提示词
            style = gr.Dropdown(['None', 'Enhance', 'Anime', 'Photographic', 'Digital Art', 'Comic Book', 'Fantasy Art', 'Analog Film', 'Cinematic', '3D Model', 'Neon Punk', 'Pixel Art', 'Isometric', 'Low Poly', 'Origami', 'Line Art', 'Craft Clay', 'Texture'], value='None', label='风格')# 创建一个下拉菜单，供用户选择风格，默认None
    # 创建第二行
    with gr.Row():
                height = gr.Slider(512, 1024, 768, step=128, label='高度')# 设置图像高度的滑块
                width = gr.Slider(512, 1024, 768, step=128, label='宽度') # 设置图像宽度的滑块
    # 创建第三行
    with gr.Row():
                scale = gr.Slider(1, 15, 10, step=.25, label='引导系数')  # 设置引导系数的滑块
                steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数') # 设置迭代步数的滑块
            seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子') # 设置随机数种子的滑块
    # 创建第四行
    with gr.Row():
                    clear = gr.Button("清除🧹") # 清除按钮
                    submit = gr.Button("提交🚀")# 提交按钮
    # 创建第五行
    with gr.Column(scale=3):
            output_image = gr.Image() # 创建用于显示图像的组件

    submit.click(fn=display_pipeline, inputs=[prompt, negative_prompt, style, height, width, scale, steps, seed], outputs=output_image) #点击提交按钮时执行display_pipeline函数
    clear.click(fn=clear_fn, inputs=clear, outputs=[prompt, negative_prompt, style, height, width, scale, steps, output_image])# 点击清除按钮时执行clear_fn函数

demo.queue(status_update_rate=1).launch(share=False)# 创建Gradio界面并启动应用
