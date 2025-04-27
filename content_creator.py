
import os
import time
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (if any)
load_dotenv()

class ContentCreationModel:
    def __init__(self, use_api_for_text=False):
        """Initialize the content creation model

        Args:
            use_api_for_text (bool): Whether to use Mistral API or local model
        """
        self.use_api_for_text = use_api_for_text

        # Initialize models
        logger.info("Initializing models...")
        self.text_model = self._setup_mistral()
        self.image_model = self._setup_stable_diffusion()
        logger.info("Models initialized successfully")

    def _setup_mistral(self):
        """Set up the Mistral text generation model"""
        if self.use_api_for_text:
            # API-based setup (simpler but requires API key)
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                logger.error("MISTRAL_API_KEY not found in environment variables")
                return None

            return {"api_key": api_key}
        else:
            # Local model setup (more complex but free to use)
            try:
                logger.info("Loading Mistral model locally...")
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info("Mistral model loaded successfully")
                return {"model": model, "tokenizer": tokenizer}
            except Exception as e:
                logger.error(f"Error loading Mistral model: {e}")
                return None

    def _setup_stable_diffusion(self):
        """Set up the Stable Diffusion image generation model"""
        try:
            logger.info("Loading Stable Diffusion model...")
            model_id = "runwayml/stable-diffusion-v1-5"  # Smaller model for beginners

            # Load the model with reduced precision for efficiency
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None  # Note: Removing safety checker is not recommended for production
            )

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)

            logger.info(f"Stable Diffusion model loaded successfully on {device}")
            return pipe
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion: {e}")
            return None

    def generate_quote(self, theme=None):
        """Generate a quote or text using Mistral AI

        Args:
            theme (str, optional): Theme for the quote

        Returns:
            str: Generated quote
        """
        if theme:
            prompt = f"Generate a short, inspiring quote about {theme} that would work well for social media content. Keep it under 20 words."
        else:
            prompt = "Generate a short, inspiring quote that would work well for social media content. Keep it under 20 words."

        if self.use_api_for_text:
            # Use Mistral API
            response = self._call_mistral_api(prompt)
            return response.strip()
        else:
            # Use local model
            tokenizer = self.text_model["tokenizer"]
            model = self.text_model["model"]

            # Prepare the input with proper formatting
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)

            # Generate text
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )

            # Decode and extract the response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the quote part (remove any system or formatting text)
            # This may need adjustment based on the model's output format
            quote = full_response.split(prompt)[-1].strip()

            return quote

    def _call_mistral_api(self, prompt):
        """Call the Mistral API to generate text

        Args:
            prompt (str): Prompt for text generation

        Returns:
            str: Generated text
        """
        api_key = self.text_model["api_key"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-small",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"API call failed: {response.status_code} - {response.text}")
            return "An inspirational quote for your journey ahead."

    def generate_images(self, quote, num_images=4, style="anime"):
        """Generate images based on the quote

        Args:
            quote (str): Text to base the images on
            num_images (int): Number of images to generate
            style (str): Visual style for the images

        Returns:
            list: List of PIL Image objects
        """
        images = []

        # Create prompts for each image with slight variations
        base_prompt = f"{quote}, {style} style, high quality, vibrant colors"

        prompts = [
            f"{base_prompt}, wide shot",
            f"{base_prompt}, close-up",
            f"{base_prompt}, dramatic lighting",
            f"{base_prompt}, soft focus"
        ]

        # Generate images
        logger.info(f"Generating {num_images} images...")
        for i, prompt in enumerate(prompts):
            if i < num_images:
                try:
                    # Generate image
                    image = self.image_model(prompt, num_inference_steps=30).images[0]
                    images.append(image)
                    logger.info(f"Generated image {i+1}/{num_images}")
                except Exception as e:
                    logger.error(f"Error generating image {i+1}: {e}")

        return images

    def add_text_to_image(self, image, quote):
        """Add text overlay to an image

        Args:
            image (PIL.Image): Image to add text to
            quote (str): Text to add to the image

        Returns:
            PIL.Image: Image with text overlay
        """
        # Create a copy of the image
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Use a default font if custom font is not available
        try:
            # Adjust path or use a built-in font
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        # Break long text into multiple lines
        words = quote.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # Check if adding this word would make the line too long
            if draw.textlength(test_line, font=font) < img.width * 0.8:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Calculate text position (centered)
        text_height = len(lines) * 40  # Approximate height per line
        y_position = (img.height - text_height) // 2

        # Add semi-transparent background for better readability
        for line in lines:
            # Get line width
            line_width = draw.textlength(line, font=font)
            x_position = (img.width - line_width) // 2

            # Draw background rectangle
            rect_padding = 10
            draw.rectangle(
                [
                    x_position - rect_padding,
                    y_position - rect_padding,
                    x_position + line_width + rect_padding,
                    y_position + 30 + rect_padding
                ],
                fill=(0, 0, 0, 128)  # Semi-transparent black
            )

            # Draw text
            draw.text(
                (x_position, y_position),
                line,
                font=font,
                fill=(255, 255, 255, 255)  # White text
            )

            y_position += 40

        return img

    def create_video(self, images, quote, output_path="output_video.mp4"):
        """Create a video from the generated images with text overlay"""
        try:
            import imageio
            import moviepy.editor as mpy
            
            # Create folder if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            logger.info("Creating video from images...")
            
            # Add text overlay to images
            images_with_text = [self.add_text_to_image(img, quote) for img in images]
            
            # Create clips from images
            clips = []
            for img in images_with_text:
                # Convert PIL image to numpy array
                img_array = np.array(img)
                # Create a clip that displays for 2.5 seconds
                clip = mpy.ImageClip(img_array).set_duration(2.5)
                clips.append(clip)
            
            # Concatenate clips
            final_clip = mpy.concatenate_videoclips(clips, method="compose")
            
            # Add simple fade transitions
            final_clip = final_clip.fadein(0.5).fadeout(0.5)
            
            # Add background music (optional)
            # If you want to add background music, you would need a music file
            # audio = mpy.AudioFileClip("path_to_music.mp3")
            # audio = audio.set_duration(final_clip.duration)
            # final_clip = final_clip.set_audio(audio)
            
            # Write video file
            final_clip.write_videofile(
                output_path, 
                fps=24, 
                codec="libx264", 
                audio_codec="aac" if hasattr(final_clip, 'audio') else None,
                preset="ultrafast",  # Use "medium" for better quality but slower encoding
                threads=4
            )
            
            logger.info(f"Video created and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return None
    
    def generate_content(self, theme=None, output_path="static/videos/content.mp4"):
        """Main function to generate complete social media content

        Args:
            theme (str, optional): Theme for the content
            output_path (str): Path to save the video

        Returns:
            dict: Result containing quote and video path
        """
        try:
            # Step 1: Generate quote
            quote = self.generate_quote(theme)
            logger.info(f"Generated quote: {quote}")

            # Step 2: Generate images based on quote
            images = self.generate_images(quote)
            logger.info(f"Generated {len(images)} images")

            # If no images were generated, return error
            if not images:
                return {"error": "Failed to generate images", "quote": quote}

            # Step 3: Create video with text overlay
            video_path = self.create_video(images, quote, output_path)
            
            if not video_path:
                return {"error": "Failed to create video", "quote": quote}

            return {
                "quote": quote,
                "video_path": video_path
            }
        except Exception as e:
            logger.error(f"Error in generate_content: {e}")
            return {"error": str(e), "quote": quote}