from flask import Flask, render_template, request, jsonify, send_from_directory
from content_creator import ContentCreationModel
import os
import uuid
from pyngrok import ngrok
import logging
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Set your ngrok authtoken
ngrok.set_auth_token("2wIpkzyxu6S3TfvUOmC9NuSYZvb_3aes6FBrNBz3rsBWg6T63")

# Initialize your model
content_model = ContentCreationModel(use_api_for_text=True)

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

        if not result.get('video_path'):
            return jsonify({
                'status': 'error',
                'message': "Video generation failed",
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
    
    # Start Flask app in a thread
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel available at: {public_url}")

    app.run(port=port)
