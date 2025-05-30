from flask import Flask, render_template, request, jsonify, send_from_directory
from content_creator import ContentCreationModel
import os
import uuid
import logging
from pyngrok import ngrok
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure ngrok
def setup_ngrok():
    try:
        auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if not auth_token:
            logger.warning("No NGROK_AUTH_TOKEN environment variable found. Please set it.")
            return False
        
        ngrok.set_auth_token(auth_token)

        # Check if there's already an authtoken in the config
        result = subprocess.run(['ngrok', 'config', 'check'], capture_output=True, text=True)
        if 'authtoken:' not in result.stdout:
            logger.info("Setting ngrok authtoken...")
            subprocess.run(['ngrok', 'config', 'add-authtoken', auth_token])
            logger.info("Ngrok authtoken added successfully")
        else:
            logger.info("Ngrok authtoken already configured")
        return True
    except Exception as e:
        logger.error(f"Error setting up ngrok: {e}")
        return False

# Initialize content model (lazy loading)
content_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    global content_model

    if content_model is None:
        try:
            logger.info("Initializing ContentCreationModel...")
            content_model = ContentCreationModel(use_api_for_text=True)
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return jsonify({
                'status': 'error',
                'message': f"Failed to initialize AI models: {str(e)}",
                'quote': "Model initialization failed"
            }), 500

    data = request.json
    theme = data.get('theme', None)

    unique_id = str(uuid.uuid4())
    output_file = f"static/videos/content_{unique_id}.mp4"

    try:
        result = content_model.generate_content(theme, output_file)

        if "error" in result:
            return jsonify({
                'status': 'error',
                'message': result['error'],
                'quote': result['quote']
            }), 500

        return jsonify({
            'status': 'success',
            'quote': result['quote'],
            'video_url': f"/static/videos/{os.path.basename(result['video_path'])}"
        })
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return jsonify({
            'status': 'error',
            'message': f"Content generation failed: {str(e)}",
            'quote': "An error occurred during content generation"
        }), 500

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('static/videos', filename)

if __name__ == '__main__':
    os.makedirs('static/videos', exist_ok=True)

    if setup_ngrok():
        public_url = ngrok.connect(5000)
        logger.info(f" * ngrok tunnel URL: {public_url}")

    logger.info("Starting Flask application...")
    app.run()
