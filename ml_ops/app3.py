from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid GUI-related errors
import matplotlib.pyplot as plt
from vae_config import model  # Import the pre-trained model
import webbrowser
from threading import Timer

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML frontend

# Preprocess a single video for the model
def preprocess_single_video(video_path, sequence_size=30, image_width=112, image_height=112):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < sequence_size:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (image_width, image_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)

    cap.release()
    while len(frames) < sequence_size:
        frames.append(np.zeros((image_height, image_width, 1)))

    video_sequence = np.expand_dims(np.array(frames), axis=0)
    return video_sequence, len(frames)

# Display all frames and their corresponding heatmaps
def generate_heatmaps(video_sequence, reconstructed_frames, sequence_length, output_path):
    if reconstructed_frames.ndim == 2:
        batch_size, seq_len, height, width, channels = video_sequence.shape
        reconstructed_frames = reconstructed_frames.reshape(batch_size, seq_len, height, width, channels)

    # Set up the plot dimensions
    fig, axes = plt.subplots(2, sequence_length, figsize=(sequence_length * 3, 6))

    for frame_idx in range(sequence_length):
        original_frame = video_sequence[0, frame_idx, :, :, 0]  # Remove batch and channel dimensions
        reconstructed_frame = reconstructed_frames[0, frame_idx, :, :, 0]

        # Calculate reconstruction error for current frame
        frame_error = np.square(original_frame - reconstructed_frame)

        axes[0, frame_idx].imshow(original_frame, cmap='gray')
        axes[0, frame_idx].axis('off')

        axes[1, frame_idx].imshow(frame_error, cmap='jet')
        axes[1, frame_idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Save the uploaded video
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, file.filename)
    file.save(video_path)

    # Preprocess the video
    sequence_size = 30
    image_width, image_height = 112, 112
    video_sequence, valid_frame_count = preprocess_single_video(video_path, sequence_size, image_width, image_height)

    # Predict with the ConvVAE model
    reconstructed_frames = model.predict(video_sequence)

    # Generate heatmaps
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    heatmap_path = os.path.join(results_dir, f"{file.filename.split('.')[0]}_heatmaps.png")
    generate_heatmaps(video_sequence, reconstructed_frames, valid_frame_count, heatmap_path)

    return jsonify({
        'heatmaps': f"/results/{file.filename.split('.')[0]}_heatmaps.png"
    })

@app.route('/results/<path:path>', methods=['GET'])
def get_result(path):
    return send_file(os.path.join('results', path))

# Run the Flask app and open the browser automatically
if __name__ == '__main__':
    port = 54508  # Specify your port here
    url = f"http://127.0.0.1:{port}"

    # Open the web browser after a slight delay
    Timer(1, lambda: webbrowser.open(url)).start()

    # Run the Flask app
    app.run(debug=False, host='127.0.0.1', port=port)
