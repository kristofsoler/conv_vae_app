from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid GUI-related errors
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

# Generate heatmap video
def generate_heatmap_video(video_sequence, reconstructed_frames, sequence_length, output_path):
    """
    Generate a video combining original frames and their corresponding heatmaps.
    Saves the video to the specified output path.
    """
    if reconstructed_frames.ndim == 2:
        batch_size, seq_len, height, width, channels = video_sequence.shape
        reconstructed_frames = reconstructed_frames.reshape(batch_size, seq_len, height, width, channels)

    # Set up video writer
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10, (224, 112))  # Width = 2x112 for side-by-side frames

    for frame_idx in range(sequence_length):
        original_frame = video_sequence[0, frame_idx, :, :, 0]  # Original frame
        reconstructed_frame = reconstructed_frames[0, frame_idx, :, :, 0]  # Reconstructed frame

        # Calculate reconstruction error (heatmap)
        frame_error = np.square(original_frame - reconstructed_frame)

        # Normalize the error for visualization
        frame_error = (frame_error / frame_error.max()) * 255.0
        frame_error = frame_error.astype(np.uint8)

        # Convert frames to BGR for video
        original_frame_bgr = cv2.cvtColor((original_frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        heatmap_bgr = cv2.applyColorMap(frame_error, cv2.COLORMAP_JET)

        # Concatenate original frame and heatmap side by side
        combined_frame = np.hstack((original_frame_bgr, heatmap_bgr))

        # Write the frame to the video
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Video saved at: {output_path}")

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

    # Generate video with heatmaps
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    video_output_path = os.path.join(results_dir, f"{file.filename.split('.')[0]}_heatmap_video.mp4")
    generate_heatmap_video(video_sequence, reconstructed_frames, valid_frame_count, video_output_path)

    return jsonify({
        'video_url': f"/results/{file.filename.split('.')[0]}_heatmap_video.mp4"
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
