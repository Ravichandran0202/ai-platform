from ml_models import train_user_segmentation, train_risk_model

print("🚀 Training models...")

train_user_segmentation()
train_risk_model()

print("✅ Training completed")
