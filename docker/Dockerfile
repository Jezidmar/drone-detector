# This is a version created for drone detection model.
FROM python:3.13-slim

# Set the working directory to /opt/app
WORKDIR /opt/app

# Install packages
RUN apt-get update -qq && apt-get install -y \
    pulseaudio \
    libasound2 libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev \
    python3-pyaudio \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Add user to 'pulse-access' group
RUN adduser root pulse-access

# Install Python packages
RUN pip install gdown 
# Copy requirements file and install Python dependencies
COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt

# Download models using gdown
RUN gdown --id '1HKWBwAKEsjpLqWf7Y6a5TQTbGQDkAeMV' -O EAT_base_export_int4.onnx.data
RUN gdown --id '1t2uic1iwkQwtJvxN9NIhdSFdwuRXFv0x' -O EAT_base_export_int4.onnx


# Copy everything else
COPY . /opt/app

# Make port 80 available to the world outside this container
EXPOSE 80
RUN mkdir -p /opt/app/samples
# Define environment variables
ENV NAME=World

# Set the entrypoint script
COPY entrypoint.sh /opt/bin/entrypoint.sh
RUN chmod +x /opt/bin/entrypoint.sh
ENTRYPOINT ["/opt/bin/entrypoint.sh"]

# Set the default command
CMD ["python3", "s_infer.py", "--model_path", "EAT_base_export_int4.onnx", "--sample_rate", "16000", "--save_path", "/samples"]