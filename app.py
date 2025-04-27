from flask import Flask, render_template, request, jsonify, send_from_directory
from content_creator import ContentCreationModel
import os
import uuid

app = Flask(__name__)
# Initialize with API usage for simpler setup for beginners
content_model = ContentCreationModel(use_api_for_text=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.json
    theme = data.get('theme', None)
    
    # Create a unique filename
    unique_id = str(uuid.uuid4())
    output_file = f"static/videos/content_{unique_id}.mp4"
    
    # Generate content
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

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('static/videos', filename)

if __name__ == '__main__':
    # Make sure directories exist
    os.makedirs('static/videos', exist_ok=True)
    app.run(debug=True)