from django.db import models
from django.utils import timezone
import uuid

class DetectionResult(models.Model):
    """Model to store object detection results"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image_name = models.CharField(max_length=255)
    image_path = models.CharField(max_length=500, blank=True, null=True)
    total_objects_detected = models.IntegerField(default=0)
    average_confidence = models.FloatField(default=0.0)
    processing_time = models.FloatField(default=0.0, help_text="Processing time in seconds")
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Detection Result"
        verbose_name_plural = "Detection Results"
    
    def __str__(self):
        return f"Detection for {self.image_name} - {self.total_objects_detected} objects"

class DetectedObject(models.Model):
    """Model to store individual detected objects"""
    
    detection_result = models.ForeignKey(
        DetectionResult, 
        on_delete=models.CASCADE, 
        related_name='detected_objects'
    )
    class_name = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    bbox_x1 = models.IntegerField(help_text="Top-left x coordinate")
    bbox_y1 = models.IntegerField(help_text="Top-left y coordinate")
    bbox_x2 = models.IntegerField(help_text="Bottom-right x coordinate")
    bbox_y2 = models.IntegerField(help_text="Bottom-right y coordinate")
    
    class Meta:
        ordering = ['-confidence_score']
        verbose_name = "Detected Object"
        verbose_name_plural = "Detected Objects"
    
    def __str__(self):
        return f"{self.class_name} ({self.confidence_score:.2f}) in {self.detection_result.image_name}"
    
    @property
    def bbox_width(self):
        return self.bbox_x2 - self.bbox_x1
    
    @property
    def bbox_height(self):
        return self.bbox_y2 - self.bbox_y1
    
    @property
    def bbox_area(self):
        return self.bbox_width * self.bbox_height

class ModelMetrics(models.Model):
    """Model to store model performance metrics"""
    
    model_name = models.CharField(max_length=100, default="RCNN_Model")
    total_predictions = models.IntegerField(default=0)
    average_objects_per_image = models.FloatField(default=0.0)
    average_confidence = models.FloatField(default=0.0)
    average_processing_time = models.FloatField(default=0.0)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Model Metrics"
        verbose_name_plural = "Model Metrics"
    
    def __str__(self):
        return f"{self.model_name} - {self.total_predictions} predictions"