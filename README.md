# Signova — Sign Language Learning & Recognition Platform

![GitHub repo size](https://img.shields.io/github/repo-size/yach26/Signova_MPR_SEM_4?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/yach26/Signova_MPR_SEM_4?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/yach26/Signova_MPR_SEM_4?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/yach26/Signova_MPR_SEM_4?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/yach26/Signova_MPR_SEM_4?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?style=for-the-badge&logo=next.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge)

---

## About

A comprehensive full-stack AI platform for learning and real-time detection of sign language gestures. Features both a web-based learning interface with ASL instruction and a dynamic gesture recognition engine using LSTM neural networks.

**Tech Stack:**
- **Frontend:** Next.js 16 with TypeScript, TailwindCSS, shadcn/ui
- **Backend:** FastAPI (Python 3.10+), SQLAlchemy ORM
- **ML Models:** PyTorch (ResNet50 for ASL, LSTM for dynamic gestures)
- **Hand Tracking:** MediaPipe Hand Landmarker
- **Database:** SQLite (with async support)

---

## Project Structure

```
MPR_SEM_4/
├── backend/                        # FastAPI server
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # App entry point
│   │   ├── core/
│   │   │   ├── config.py          # Settings & environment
│   │   │   └── database.py        # SQLAlchemy setup
│   │   ├── models/                # SQLAlchemy ORM models
│   │   │   ├── user_model.py
│   │   │   ├── learning_model.py
│   │   │   └── progress_model.py
│   │   ├── routes/                # API endpoints
│   │   │   ├── auth.py            # SignUp / Login
│   │   │   ├── dashboard.py       # User dashboard
│   │   │   ├── learning.py        # Learning content
│   │   │   ├── quiz.py            # Quiz endpoints
│   │   │   ├── predict.py         # Sign prediction
│   │   │   └── progress.py        # Progress tracking
│   │   ├── schemas/               # Pydantic validation
│   │   │   ├── user_schema.py
│   │   │   └── learning_schema.py
│   │   ├── services/              # Business logic
│   │   │   ├── model_service.py   # ASL model inference
│   │   │   ├── dynamic_model_service.py  # Dynamic gesture recognition
│   │   │   └── quiz_data.py       # Quiz management
│   │   └── utils/
│   │       └── auth_utils.py      # JWT helpers
│   ├── server.py                  # Uvicorn entry point
│   ├── requirements.txt           # Python dependencies
│   ├── test_signup.py             # Test suite
│   └── fix_model.py               # Model utilities
│
├── frontend/                       # Next.js 16 application
│   ├── app/
│   │   ├── layout.tsx             # Root layout
│   │   ├── page.tsx               # Home page
│   │   ├── globals.css
│   │   ├── dashboard/             # Dashboard pages
│   │   ├── detection/             # Sign detection UI
│   │   ├── learning/              # Learning pages
│   │   └── quiz/                  # Quiz interface
│   ├── components/
│   │   ├── auth-modal.tsx
│   │   ├── navbar.tsx
│   │   ├── footer.tsx
│   │   ├── theme-provider.tsx
│   │   ├── home/
│   │   ├── learning/
│   │   └── ui/                    # shadcn/ui components
│   ├── contexts/
│   │   └── auth-context.tsx       # Authentication context
│   ├── hooks/
│   │   ├── use-mobile.ts
│   │   └── use-toast.ts
│   ├── lib/
│   │   └── utils.ts               # Utility functions
│   ├── package.json
│   ├── tsconfig.json
│   └── next.config.mjs
│
├── Dynamic_M/                      # LSTM-based dynamic gesture recognition
│   ├── app.py                     # FastAPI server for dynamic model
│   ├── train.py                   # Model training pipeline
│   ├── realtime.py                # Real-time inference with OpenCV
│   ├── model.py                   # LSTM architecture
│   ├── inference_utils.py         # Inference utilities
│   ├── dataset_loader.py          # Dataset loading & augmentation
│   ├── collect_data.py            # Interactive data collection
│   ├── model_loader.py            # Model singleton registry
│   ├── utils.py                   # Helper functions
│   ├── labels.json                # Gesture class labels
│   ├── dataset/                   # Training data (collected per gesture)
│   │   ├── hello/
│   │   ├── bye/
│   │   ├── yes/
│   │   ├── no/
│   │   ├── ily/
│   │   ├── thank_you/
│   │   ├── sorry/
│   │   └── please/
│   ├── models/                    # Trained model checkpoints
│   │   └── training_config.json
│   ├── mnt/
│   │   └── user-data/
│   └── requirements.txt
│
├── MPR_STATIC_M/                   # Static ASL model utilities
│   ├── local_inference.py         # Standalone inference script
│   ├── pretrained_mirror.py       # Mirror-augmented inference
│   ├── asl_classes.json           # ASL class labels
│   └── requirements.txt
│
└── README.md                       # This file
```

---

## Features

### 🎓 Learning Platform
- **User Authentication:** Secure signup/login with JWT tokens
- **Interactive Dashboard:** Progress tracking and learning statistics
- **Structured Learning:** Organized ASL lessons and tutorials
- **Quiz System:** Test knowledge on learned signs
- **Real-time Detection:** Capture and identify sign language in real-time

### 🚀 Dynamic Gesture Recognition
- **LSTM Classification:** 8 gesture classes (hello, bye, yes, no, ily, thank_you, sorry, please)
- **Real-time Processing:** 30-frame sequence buffering for accuracy
- **MediaPipe Integration:** 21-point hand landmark detection
- **REST API:** FastAPI endpoints for prediction and batch processing
- **Data Collection:** Interactive tool for gathering new gesture data
- **Training Pipeline:** Complete training with validation, loss curves, and confusion matrices

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm/pnpm | Latest |
| Git | Latest |

### 1. Clone Repository

```bash
git clone https://github.com/yach26/Signova_MPR_SEM_4.git
cd Signova_MPR_SEM_4
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set SECRET_KEY to a random string

# Start server
python server.py
```

Backend runs at **http://localhost:8000**  
API docs: **http://localhost:8000/docs**  
ReDoc: **http://localhost:8000/redoc**

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# Default API URL: http://localhost:8000 (no change needed for local dev)

# Start dev server
npm run dev
```

Frontend runs at **http://localhost:3000**

### 4. (Optional) Dynamic Gesture Recognition

For standalone gesture recognition training and inference:

```bash
cd Dynamic_M

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Collect training data
python collect_data.py

# Train the model
python train.py --dataset ./dataset --output ./models --epochs 100

# Run real-time inference
python realtime.py --model ./models/best_model.pth --config ./models/training_config.json

# Or start FastAPI server
python app.py
```

---

## API Documentation

### Learning Platform Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/signup` | Register new user |
| `POST` | `/auth/login` | Login & get JWT token |
| `GET` | `/dashboard` | Get user dashboard data |
| `GET` | `/learning` | Get learning content |
| `GET` | `/quiz/questions` | Fetch quiz questions |
| `POST` | `/quiz/submit` | Submit quiz answers |
| `GET` | `/progress` | Get user progress stats |
| `POST` | `/predict` | Predict sign from image |
| `POST` | `/predict/base64` | Predict sign from base64 image |
| `GET` | `/model/status` | Check model load status |

### Dynamic Model Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server & model status |
| `GET` | `/classes` | List gesture classes |
| `POST` | `/predict` | Predict single sequence |
| `POST` | `/predict/batch` | Batch prediction |

---

## How It Works

### ASL Detection Pipeline

```
Browser Camera Frame
  ↓
Canvas Capture (JPEG)
  ↓
POST /predict (multipart)
  ↓
MediaPipe HandLandmarker (21 × 3 landmarks)
  ↓
Normalize Landmarks (origin=wrist, scale=middle finger)
  ↓
ResNet50 Inference (with mirror augmentation)
  ↓
Return Prediction: { "label": "A", "confidence": 0.97 }
  ↓
Display in UI
```

### Dynamic Gesture Recognition Pipeline

```
Webcam Input
  ↓
MediaPipe Hand Landmarks (63 features/frame)
  ↓
30-Frame Sequence Buffer
  ↓
LSTM Network (128 hidden, 2 layers)
  ↓
Softmax Classification
  ↓
Predicted Gesture + Confidence Score
```

---

## Environment Variables

### Backend (`backend/.env`)

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./signlang.db

# JWT Authentication
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# CORS (if needed)
ALLOWED_ORIGINS=http://localhost:3000
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Model Architecture

### ASL ResNet50 Model
- **Input:** Image with hand
- **Extractor:** ResNet50 backbone
- **Output:** 28 ASL classes (A-Z, delete, space)
- **Optimization:** Mirror-augmented predictions for both hands
- **Device:** CPU/CUDA auto-detection

### Dynamic LSTM Model
- **Input:** (batch, 30, 63) — 30 frames × 63 features
- **Architecture:** 
  - LSTM Layer 1: 128 hidden units
  - LSTM Layer 2: 128 hidden units
  - Dropout: 0.3
  - Classifier: 128 → 64 → ReLU → num_classes
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **LR Schedule:** ReduceLROnPlateau (factor=0.5, patience=7)

---

## Key Dependencies

### Backend
- `fastapi==0.111.0` — Web framework
- `uvicorn` — ASGI server
- `torch==2.4.1` — Deep learning
- `torchvision==0.19.1` — Computer vision
- `mediapipe` — Hand tracking
- `sqlalchemy` + `aiosqlite` — Async database
- `pydantic` v2 — Data validation
- `bcrypt` + `PyJWT` — Security

### Frontend
- `next@latest` — React framework
- `react@19` — UI library
- `typescript` — Type safety
- `tailwindcss@4` — Styling
- `shadcn/ui` — Component library
- `framer-motion` — Animations
- `lucide-react` — Icons

---

## Data Collection

### Gesture Data Collection (Dynamic_M)

```bash
cd Dynamic_M
python collect_data.py
```

**Controls:**
- Camera feed with hand landmark visualization
- 3-second countdown before recording
- Collects 30 sequences × 30 frames per gesture
- Data saved as `.npy` files to `dataset/<gesture>/`
- Press `Q` to quit, `S` to skip current gesture

### Training

```bash
python train.py \
    --dataset ./dataset \
    --output ./models \
    --epochs 100 \
    --batch 32 \
    --lr 0.001 \
    --patience 15
```

**Outputs:**
- `best_model.pth` — Best model checkpoint
- `training_config.json` — Hyperparameters
- `training_curves.png` — Loss/accuracy plots
- `confusion_matrix.png` — Per-class metrics

---

## Troubleshooting

### Backend Issues

**Model not loading:**
```bash
# Check model status
curl http://localhost:8000/model/status

# Verify dependencies
pip install -r requirements.txt
```

**Database errors:**
```bash
# Remove old database and recreate
rm signlang.db
python server.py  # Recreates schema
```

### Frontend Issues

**API connection errors:**
- Verify `NEXT_PUBLIC_API_URL` matches backend URL
- Ensure backend is running: `http://localhost:8000/health`
- Check browser console for CORS errors

**Module not found:**
```bash
# Reinstall dependencies
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

---

## Performance Tips

1. **GPU Acceleration:** Ensure CUDA is installed for PyTorch
2. **Database Indexing:** Add indices on frequently queried fields
3. **Caching:** Frontend caches auth tokens in localStorage
4. **Batch Processing:** Use `/predict/batch` for multiple sequences
5. **Model Quantization:** Consider converting to ONNX for production

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 (Python) and ESLint (TypeScript)
- Tests pass locally
- Documentation is updated

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **MediaPipe:** Hand landmark detection framework
- **PyTorch:** Deep learning framework
- **Next.js:** React framework
- **FastAPI:** Async Python web framework
- **shadcn/ui:** Component library
- ASL community for gestures and feedback

---

## Contact & Support

For issues, questions, or suggestions:
- 📧 Open an issue on GitHub
- 🔗 Visit the repository: https://github.com/yach26/Signova_MPR_SEM_4

**Made with ❤️ for the deaf and hard-of-hearing community**

---

## Tips for Better Accuracy

1. **Lighting** — collect data in consistent lighting; avoid harsh shadows on hands
2. **Hand distance** — stay 30–60 cm from camera; keypoints normalize position automatically
3. **Variation** — collect data from multiple angles and speeds
4. **More data** — increase `NUM_SEQUENCES` in `collect_data.py` (default 30 → try 50+)
5. **Augmentation** — already applied in `dataset_loader.py` (noise, flip, scale, shift)
