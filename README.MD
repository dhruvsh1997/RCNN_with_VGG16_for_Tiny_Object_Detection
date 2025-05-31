# 🎯 Django RCNN Object Detection Web Application

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Django](https://img.shields.io/badge/Django-4.2+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

*A production-ready Django web application for real-time object detection using Region-based Convolutional Neural Networks (R-CNN)*

[🚀 Quick Start](#-quick-start) •
[📸 Demo](#-demo) •
[🏗️ Architecture](#️-architecture) •
[🐳 Docker](#-docker-deployment) •
[📚 Documentation](#-api-documentation)

</div>

---

## 🌟 Overview

This Django web application provides a seamless interface for object detection using a trained R-CNN model. Users can upload images through a modern drag-and-drop interface and receive real-time object detection results with bounding boxes, confidence scores, and detailed analytics.

<div align="center">

### 🎥 Demo

![Object Detection Demo](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Upload+Image+→+AI+Detection+→+Visual+Results)

</div>

## ✨ Key Features

🖼️ **Smart Image Processing**
- Drag & drop image upload with instant preview
- Support for multiple image formats (JPEG, PNG, WebP)
- Automatic image preprocessing and optimization
- Real-time progress indicators

🤖 **Advanced AI Detection**
- R-CNN based object detection with VGG16 backbone
- Customizable confidence thresholds
- Non-Maximum Suppression for clean results
- Support for multiple object classes

🎨 **Modern User Interface**
- Responsive Bootstrap 5 design
- Animated loading states and transitions
- Side-by-side result comparison
- Mobile-friendly responsive layout

📊 **Analytics & Insights**
- Detection confidence scores
- Bounding box coordinates
- Processing time metrics
- Database logging for analytics

🔧 **Production Ready**
- Docker containerization
- Comprehensive error handling
- Security best practices
- Scalable architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- 4GB+ RAM recommended

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/django-rcnn-detection.git
cd django-rcnn-detection
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv envRCNNObj

# Activate virtual environment
# On Windows:
envRCNNObj\Scripts\activate
# On macOS/Linux:
source envRCNNObj/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Trained Model

```bash
# Copy your trained R-CNN model to the media directory
cp /path/to/your/improved_rcnn_model.h5 media/
```

### 5. Configure Environment

Create a `.env` file in the project root:

```env
SECRET_KEY=your-super-secret-django-key-here
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
```

### 6. Database Setup

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser  # Optional: create admin user
```

### 7. Run the Application

```bash
python manage.py runserver
```

🎉 **Success!** Visit `http://127.0.0.1:8000/` to start detecting objects!

## 🏗️ Project Architecture

```
object_detection_project/
├── 📁 detection/                    # Main Django app
│   ├── 🐍 views.py                 # Core detection logic
│   ├── 🔗 urls.py                  # URL routing
│   ├── 📊 models.py                # Database models
│   └── 🧪 tests.py                 # Unit tests
├── 📁 templates/detection/          # HTML templates
│   └── 🎨 index.html              # Main interface
├── 📁 media/                       # Media files
│   ├── 🤖 improved_rcnn_model.h5  # Your trained model
│   └── 📁 temp/                   # Temporary uploads
├── 📁 static/                      # Static assets
├── 📁 logs/                        # Application logs
├── 📁 Notebook/                    # Jupyter notebooks
│   └── 📔 ObjectDetectionRCNN.ipynb
├── 🐳 Dockerfile                   # Docker configuration
├── 📋 requirements.txt             # Python dependencies
├── ⚙️ manage.py                   # Django management
└── 📄 README.md                   # This file
```

### 🧠 AI Model Architecture

```mermaid
graph TD
    A[📷 Input Image] --> B[🔍 Selective Search]
    B --> C[📦 Region Proposals]
    C --> D[✂️ ROI Extraction]
    D --> E[🧠 VGG16 Backbone]
    E --> F[🗺️ Feature Maps]
    F --> G[🏷️ Classification Head]
    F --> H[📐 Regression Head]
    G --> I[🎯 Object Classes]
    H --> J[📦 Bounding Boxes]
    I --> K[🔄 Non-Max Suppression]
    J --> K
    K --> L[✅ Final Detections]
```

## 🎨 User Interface

### Main Features

| Feature | Description |
|---------|-------------|
| **Drag & Drop Upload** | Intuitive image upload with visual feedback |
| **Real-time Processing** | Live progress indicators during detection |
| **Visual Results** | Side-by-side comparison with bounding boxes |
| **Detection Analytics** | Confidence scores and processing metrics |
| **Responsive Design** | Works seamlessly on desktop and mobile |

### Screenshots

<div align="center">

| Upload Interface | Detection Results |
|------------------|-------------------|
| ![Upload](https://via.placeholder.com/400x300/2196F3/FFFFFF?text=Drag+%26+Drop+Upload) | ![Results](https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Detection+Results) |

</div>

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build the Docker image
docker build -t django-rcnn-app .

# Run the container
docker run -p 8000:8000 -v $(pwd)/media:/app/media django-rcnn-app
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./media:/app/media
      - ./logs:/app/logs
    environment:
      - DEBUG=False
      - SECRET_KEY=${SECRET_KEY}
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## ⚙️ Configuration

### Model Configuration

Update the categories in `detection/views.py` to match your trained model:

```python
# Update this dictionary with your actual model classes
categories = {
    0: 'background',
    1: 'person',
    2: 'car', 
    3: 'bicycle',
    4: 'motorcycle',
    # Add your custom classes here
}
```

### Performance Tuning

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.3 | Minimum confidence for detections |
| `NMS_THRESHOLD` | 0.3 | Non-Maximum Suppression threshold |
| `FILE_UPLOAD_MAX_SIZE` | 10MB | Maximum upload file size |
| `MAX_DETECTIONS` | 100 | Maximum detections per image |

## 📊 API Documentation

### Detection Endpoint

**POST** `/detect/`

**Request:**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:8000/detect/
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300]
    }
  ],
  "processing_time": 1.23,
  "image_url": "/media/temp/processed_image.jpg"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python manage.py test

# Run specific test module
python manage.py test detection.tests

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

## 📈 Performance Metrics

### Benchmark Results

| Metric | Value |
|--------|-------|
| **Average Processing Time** | 1.2s per image |
| **Model Accuracy** | 78% validation accuracy |
| **Supported Image Formats** | JPEG, PNG, WebP |
| **Max Image Size** | 10MB |
| **Concurrent Users** | 50+ (with proper scaling) |

## 🔧 Troubleshooting

### Common Issues & Solutions

<details>
<summary><strong>🚨 Model not found error</strong></summary>

**Problem:** `FileNotFoundError: improved_rcnn_model.h5 not found`

**Solution:**
```bash
# Ensure your model file is in the correct location
ls media/improved_rcnn_model.h5

# If missing, copy your trained model:
cp /path/to/your/model.h5 media/improved_rcnn_model.h5
```
</details>

<details>
<summary><strong>🚨 OpenCV installation issues</strong></summary>

**Problem:** OpenCV import errors or missing functionality

**Solution:**
```bash
# Uninstall existing OpenCV packages
pip uninstall opencv-python opencv-contrib-python

# Install the contrib version
pip install opencv-contrib-python==4.8.1.78
```
</details>

<details>
<summary><strong>🚨 Memory issues during detection</strong></summary>

**Problem:** Out of memory errors during processing

**Solution:**
- Reduce image size before processing
- Increase Docker memory limits
- Consider using GPU acceleration
</details>

<details>
<summary><strong>🚨 File upload fails</strong></summary>

**Problem:** Large files fail to upload

**Solution:**
```python
# In settings.py, increase upload limits:
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
```
</details>

## 🚀 Production Deployment

### Security Checklist

- [ ] Set `DEBUG=False` in production
- [ ] Use environment variables for sensitive settings
- [ ] Configure HTTPS with proper SSL certificates
- [ ] Set up proper database (PostgreSQL recommended)
- [ ] Configure reverse proxy (Nginx/Apache)
- [ ] Set up monitoring and logging
- [ ] Configure backup strategies

### Scaling Considerations

- **Load Balancing:** Use multiple Django instances behind a load balancer
- **Caching:** Implement Redis for session and query caching
- **Database:** Use PostgreSQL with connection pooling
- **File Storage:** Consider cloud storage (AWS S3, Google Cloud)
- **GPU Acceleration:** Use CUDA-enabled containers for faster inference

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/django-rcnn-detection.git
cd django-rcnn-detection

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python manage.py test
```

## 📚 Resources & References

### Documentation
- [Django Documentation](https://docs.djangoproject.com/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

### Research Papers
- [R-CNN: Rich feature hierarchies for accurate object detection](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN: Towards Real-Time Object Detection](https://arxiv.org/abs/1506.01497)

### Model Training Resources
- [COCO Dataset](https://cocodataset.org/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Object Detection Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the amazing deep learning framework
- **Django Community** for the robust web framework
- **OpenCV Contributors** for computer vision utilities
- **Bootstrap Team** for the responsive UI framework
- **Contributors** who help improve this project

## 📞 Support

- 📧 **Email:** [your-email@example.com](mailto:your-email@example.com)
- 🐛 **Issues:** [GitHub Issues](https://github.com/your-username/django-rcnn-detection/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/your-username/django-rcnn-detection/discussions)
- 📖 **Wiki:** [Project Wiki](https://github.com/your-username/django-rcnn-detection/wiki)

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=your-username.django-rcnn-detection)
[![GitHub stars](https://img.shields.io/github/stars/your-username/django-rcnn-detection.svg?style=social&label=Star)](https://github.com/your-username/django-rcnn-detection)
[![GitHub forks](https://img.shields.io/github/forks/your-username/django-rcnn-detection.svg?style=social&label=Fork)](https://github.com/your-username/django-rcnn-detection/fork)

**Made with ❤️ by [Your Name](https://github.com/your-username)**

</div>