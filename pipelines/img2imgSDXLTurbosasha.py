from diffusers import (
    AutoPipelineForImage2Image,
    AutoencoderTiny,
)
from compel import Compel, ReturnedEmbeddingsType
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
from openai import OpenAI
import os
from io import BytesIO
from dotenv import load_dotenv
import base64
import asyncio

load_dotenv()

base_model = "stabilityai/sdxl-turbo"
taesd_model = "madebyollin/taesdxl"

default_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """"""

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY")
)

if not client.api_key.startswith("sk-"):
    # throw error 'Please enter your OpenAI API key!'
    raise ValueError("Please enter your OpenAI API key!") 


def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")

def get_sys_promt(use_gptv):
    what_users_will_say=[
        ("make me look like an old man", "a close up image of an old man"),
        ("more beautiful", "a close up image of a beautiful person") ,
        ("make me stand below a street light", "a close up image of a person standing below a street light"),
        ("I want to look like a dog", "a close up image of a person in a dog costume"),
        ("A person in a cat costume", "a close up image of a person in a cat costume"),
        ("Why not be an native indian", "a close up image of an native indian"),
        ("So what this also shows is that you can say you want to be a cat, and it will generate a dog eating meat for you", "a close up image of a person impersonating a cat"),
        ("It will generate a cat with your face", "a close up image of a person in a cat costume"),
    ]
    
    things_to_capture = '\n'.join([t[0] for t in what_users_will_say])
    correct_things = '\n'.join([t[1] for t in what_users_will_say])
    prompt_for_GPT = f"Generate a Stable Diffusion prompt that captures things like {things_to_capture} into {correct_things}"
    
    prompt_for_GPTV = """
    You are a GPT Vision bot. Take an image of a person and use the image description to generate a prompt for stable diffusion. For example change the prompt from:
    "I think you can just say things like 'why not make me a santa clause'" to "'a close up image of a woman in light makeup in a santa clause costume"
    with "a woman in light makeup" being the description of the image. Only output the stable diffusion prompt.
    """
    sys_prompt = prompt_for_GPTV if use_gptv else prompt_for_GPT
    return sys_prompt

async def async_wrapper(abc, *args, **kwargs):
    return await asyncio.to_thread(abc, *args, **kwargs)

# call openai to generate a response using chatCompletion api
def call_openai(system_prompt, input_text, model="gpt-4-1106-preview"):
    completion = client.chat.completions.create(
        # Use GPT 3.5 as the LLM
        temperature=0.7,
        model=model,
        # Pre-define conversation messages for the possible roles
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
    )
    return completion.choices[0].message.content
    
def call_openai_gptv(system_prompt, input_text, image_base64, model='gpt-4-vision-preview'):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                },
            ],
        },
        {
           "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
    )
    reply = response.choices[0].message.content
    return reply
    
class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image SDXL"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            hide=True,
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            4, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.2,
            min=0,
            max=20,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        if args.safety_checker:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(base_model)
        else:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                base_model,
                safety_checker=None,
            )
        if args.use_taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        # check if computer has less than 64GB of RAM using sys or os
        if psutil.virtual_memory().total < 64 * 1024**3:
            self.pipe.enable_attention_slicing()

        if args.torch_compile:
            print("Running torch compile")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )

            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (768, 768))],
            )

        self.pipe.compel_proc = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        self.old_prompt = ""


    async def alter_prompt(self, params):
        import time
        print(f"altering prompt")
        prompt = params.prompt
        use_gptv = params.use_gptv if hasattr(params, 'use_gptv') else False
        image_base64 = pil_to_base64(params.image)
        sys_prompt = get_sys_promt(use_gptv)
        print(f"useGPTV {use_gptv}")
        start_time = time.time()
        if use_gptv:
            output = await async_wrapper(call_openai_gptv, sys_prompt, prompt, image_base64=image_base64)
        else:
            output = await async_wrapper(call_openai, sys_prompt, prompt)
            
        print(f"duration for calling GPTV/{use_gptv}: {time.time() - start_time}")
        params.prompt = output
    
    async def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        if self.old_prompt != params.prompt:        
            self.old_prompt = params.prompt
            asyncio.create_task(self.alter_prompt(params))
            
        print(f"running...")  
        prompt_embeds, pooled_prompt_embeds = self.pipe.compel_proc(
            [params.prompt, params.negative_prompt]
        )
        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        results = self.pipe(
            image=params.image,
            prompt_embeds=prompt_embeds[0:1],
            pooled_prompt_embeds=pooled_prompt_embeds[0:1],
            negative_prompt_embeds=prompt_embeds[1:2],
            negative_pooled_prompt_embeds=pooled_prompt_embeds[1:2],
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
        )

        nsfw_content_detected = (
            results.nsfw_content_detected[0]
            if "nsfw_content_detected" in results
            else False
        )
        if nsfw_content_detected:
            return None
        result_image = results.images[0]

        return result_image
